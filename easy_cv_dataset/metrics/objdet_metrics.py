import keras
from keras import ops
from .non_max_supression import NonMaxSuppression
from keras.utils.bounding_boxes import compute_iou

def one_hot(y, num_classes):
    y = ops.cast(y, "int32")
    return ops.cast(ops.equal(
        ops.expand_dims(y, axis=-1),
        ops.arange(num_classes, dtype="int32")
    ), "float32")

class EvaluateMAPmetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, iou_threshold=0.5, prefix='val', num_thresholds=200):
        super().__init__()
        self.data = data
        self.prefix = prefix
        self.metric = keras.metrics.AUC(
            from_logits=False, 
            multi_label=True,
            num_thresholds=num_thresholds,
            curve="PR",
            name='mAP'
        )
        self.metric.iou_threshold = iou_threshold

    def set_model(self, model):
        super().set_model(model)
        self.model.prediction_decoder = NonMaxSuppression(
            bounding_box_format = self.model.prediction_decoder.bounding_box_format,
            from_logits = self.model.prediction_decoder.from_logits,
            iou_threshold = self.model.prediction_decoder.iou_threshold,
            confidence_threshold = self.model.prediction_decoder.confidence_threshold,
            max_detections = self.model.prediction_decoder.max_detections,
        )
        assert self.model.bounding_box_format==self.model.prediction_decoder.bounding_box_format
        self.model.compute_metrics = self._fun_compute_metrics

    def _fun_compute_metrics(self, x, y, y_pred, sample_weight=None):
        if self.model._compile_metrics is not None:
            box_pred = y_pred["bbox_regression"]
            cls_pred = y_pred["cls_logits"]
            num_classes = self.model.num_classes
            iou_threshold = self.model._compile_metrics.iou_threshold

            # box_pred is on "center_yxhw" format, convert to target format.
            if isinstance(x, list) or isinstance(x, tuple):
                images, _ = x
            else:
                images = x
            height, width, channels = ops.shape(images)[1:]
            anchor_boxes = self.model.anchor_generator(images)
            anchor_boxes = ops.concatenate(list(anchor_boxes.values()), axis=0)
            box_pred = keras.utils.bounding_boxes.decode_deltas_to_boxes(
                anchors=anchor_boxes,
                boxes_delta=box_pred,
                encoded_format="center_xywh",
                anchor_format=self.model.anchor_generator.bounding_box_format,
                box_format=self.model.bounding_box_format,
                image_shape=(height, width, channels),
            )
            y_pred = self.model.prediction_decoder(box_pred, cls_pred, images=images, flag_confidence_threshold=False)
            y_true = y

            # predicted classes
            class_pred = one_hot(y_pred['labels'], num_classes) * y_pred['confidence'][:,:,None] # B X Np X C
        
            # matching boxes # B X Np X Nt
            detect = ops.logical_and(compute_iou(
                    y_pred['boxes'],
                    y_true['boxes'],
                    bounding_box_format=self.model.bounding_box_format,
            ) >= iou_threshold, ops.equal(y_pred['labels'][:,:,None], y_true['labels'][:,None,:]))

            # detect missing boxes
            missed_boxes = ops.any(detect, axis=1)==False
            missed_boxes = ops.where(missed_boxes, y_true['labels'], -1)
            missed_boxes = one_hot(missed_boxes, num_classes)
            #print('missed_boxes:', missed_boxes)
        
            # take only the detection with max score
            mask_other = y_pred['confidence'][:,:,None]*detect  # B X Np X Nt
            mask_other = mask_other==ops.max(mask_other, axis=1, keepdims=True)
            detect = ops.logical_and(detect, mask_other)
            #print('detect:', detect)

            # convert detection to labeling
            class_true = one_hot(y_true['labels'], num_classes)
            class_true = ops.matmul(ops.cast(detect, class_true.dtype), class_true)
            #print('class_true:', class_true)

            class_true = ops.concatenate((class_true, missed_boxes), axis=1) 
            class_pred = ops.concatenate((class_pred, 0.0*missed_boxes), axis=1)
            class_true = ops.reshape(class_true, (-1, num_classes))
            class_pred = ops.reshape(class_pred, (-1, num_classes))

            self.model._compile_metrics.update_state(class_true, class_pred)
        return self.model.get_metrics_result()

    def evaluate(self, verbose=True):
        old_metrics = self.model._compile_metrics
        self.model._compile_metrics = self.metric
        try:
            results = self.model.evaluate(self.data, verbose=verbose, return_dict=True)
        except:
            self.model._compile_metrics = old_metrics
            raise
        
        self.model._compile_metrics = old_metrics
        ap = interpolate_pr_each_classes(self.metric)
        if hasattr(self.data, "class_names"):
            class_names = self.data.class_names
        else:
            class_names = list(range(self.model.num_classes))
        ap = {class_names[index]: ap[index] for index in range(len(ap))}
        return results, ap
    
    def on_epoch_end(self, epoch, logs):
        results, ap = self.evaluate(verbose=True)
        for k in results:
            logs[self.prefix+'_'+k] = results[k]
        
        print()
        for k in ap:
            print(" AP of %20s = %7.5f" % (k, ap[k]))
        
        return logs

def compute_mAP_metrics(
    model, dataset, iou_threshold=0.5, num_thresholds=200, verbose=True
):
    fun = EvaluateMAPmetricsCallback(dataset, iou_threshold=iou_threshold, num_thresholds=num_thresholds, prefix='test')
    fun.set_model(model)
    results, ap = fun.evaluate(verbose=verbose)
    for k in ap:
        results['AP_%s'%k] = ap[k].numpy()
    return results



def interpolate_pr_each_classes(metric):
    """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

    https://www.biostat.wisc.edu/~page/rocpr.pdf

    Note here we derive & use a closed formula not present in the paper
    as follows:

        Precision = TP / (TP + FP) = TP / P

    Modeling all of TP (true positive), FP (false positive) and their sum
    P = TP + FP (predicted positive) as varying linearly within each
    interval [A, B] between successive thresholds, we get

        Precision slope = dTP / dP
                        = (TP_B - TP_A) / (P_B - P_A)
                        = (TP - TP_A) / (P - P_A)
        Precision = (TP_A + slope * (P - P_A)) / P

    The area within the interval is (slope / total_pos_weight) times

        int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
        int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}

    where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in

        int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)

    Bringing back the factor (slope / total_pos_weight) we'd put aside, we
    get

        slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight

    where dTP == TP_B - TP_A.

    Note that when P_A == 0 the above calculation simplifies into

        int_A^B{Precision.dTP} = int_A^B{slope * dTP}
                                = slope * (TP_B - TP_A)

    which is really equivalent to imputing constant precision throughout the
    first bucket having >0 true positives.

    Returns:
        pr_auc: an approximation of the area under the P-R curve for each classes
    """

    dtp = ops.subtract(
        metric.true_positives[: metric.num_thresholds - 1],
        metric.true_positives[1:],
    )
    p = ops.add(metric.true_positives, metric.false_positives)
    dp = ops.subtract(p[: metric.num_thresholds - 1], p[1:])
    prec_slope = ops.divide_no_nan(dtp, ops.maximum(dp, 0))
    intercept = ops.subtract(
        metric.true_positives[1:], ops.multiply(prec_slope, p[1:])
    )

    safe_p_ratio = ops.where(
        ops.logical_and(p[: metric.num_thresholds - 1] > 0, p[1:] > 0),
        ops.divide_no_nan(
            p[: metric.num_thresholds - 1], ops.maximum(p[1:], 0)
        ),
        ops.ones_like(p[1:]),
    )

    pr_auc_increment = ops.divide_no_nan(
        ops.multiply(
            prec_slope,
            (ops.add(dtp, ops.multiply(intercept, ops.log(safe_p_ratio)))),
        ),
        ops.maximum(
            ops.add(metric.true_positives[1:], metric.false_negatives[1:]), 0
        ),
    )

    return ops.sum(pr_auc_increment, axis=0)
    