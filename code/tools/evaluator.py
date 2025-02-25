# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import numpy as np
from scipy.special import expit

import torch
import torchmetrics

def evaluator_v6(pred, label):
    num_classes = pred.shape[1]
    task = 'binary' if num_classes == 2 else 'multiclass'  # 根据类别数量确定任务类型

    # 创建准确率指标对象（整体准确率）
    accuracy_metric = torchmetrics.Accuracy(average='macro', task=task, num_classes=num_classes)
    # 创建精确率指标对象（宏平均）
    precision_metric_macro = torchmetrics.Precision(average='macro', task=task, num_classes=num_classes)
    # 创建召回率指标对象（宏平均）
    recall_metric_macro = torchmetrics.Recall(average='macro', task=task, num_classes=num_classes)
    # 创建F1 - score指标对象（宏平均）
    f1_metric_macro = torchmetrics.F1Score(average='macro', task=task, num_classes=num_classes)
    accuracy_macro = accuracy_metric(pred.argmax(dim=1), label)
    precision_macro = precision_metric_macro(pred.argmax(dim=1), label)
    recall_macro = recall_metric_macro(pred.argmax(dim=1), label)
    f1_macro = f1_metric_macro(pred.argmax(dim=1), label)
    return {
        'Accuracy': accuracy_macro.item(),
        'Precision': precision_macro.item(),
        'Recall': recall_macro.item(),
        'F1': f1_macro.item(),
        'pred':pred.to("cpu").numpy(),
        'label':label.to("cpu").numpy()
    }
def evaluator_v5(pred, label):
    """
       用于二分类任务的评估函数，计算准确率、精确率、召回率、F1值、AUC、特异度、敏感度等指标。

       参数:
       - pred (torch.Tensor): 模型的预测结果，形状应为 (batch_size, 2)，经过softmax处理后，第二维表示每个类别的概率。
       - label (torch.Tensor): 真实标签，形状应为 (batch_size,)，元素取值为0或1，表示两个类别。

       返回:
       - 包含各项评估指标的字典，包括 'Accuracy'（准确率）、'Precision'（精确率）、
         'Recall'（召回率）、'F1'（F1值）、'AUC'（曲线下面积）、'Sensitivity'（敏感度）、
         'Specificity'（特异度）、'pred'（预测标签）、'label'（真实标签）。
       """
    assert pred.shape[1] == 2, "输入的预测结果应为二分类格式"
    task = 'binary'

    # 创建准确率指标对象（整体准确率）
    accuracy_metric = torchmetrics.Accuracy(task=task)
    # 创建精确率指标对象（二分类下不需要指定平均方式）
    precision_metric = torchmetrics.Precision(task=task)
    # 创建召回率指标对象（二分类下不需要指定平均方式）
    recall_metric = torchmetrics.Recall(task=task)
    # 创建F1 - score指标对象（二分类下不需要指定平均方式）
    f1_metric = torchmetrics.F1Score(task=task)
    # 创建用于计算AUC的指标对象
    auc_value = roc_auc_score(label, pred[:, 1])


    # 计算整体的指标
    accuracy = accuracy_metric(pred.argmax(dim=1), label)
    precision = precision_metric(pred.argmax(dim=1), label)
    recall = recall_metric(pred.argmax(dim=1), label)
    f1 = f1_metric(pred.argmax(dim=1), label)

    pred_label = pred.argmax(dim=1)
    tn, fp, fn, tp = confusion_matrix(label.cpu().numpy(), pred_label.cpu().numpy()).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)


    return {
        'Accuracy': accuracy.item(),
        'Precision': precision.item(),
        'Recall': recall.item(),
        'F1': f1.item(),
        'AUC': auc_value.item(),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'pred': pred_label.to("cpu").numpy(),
        'label': label.to("cpu").numpy()
    }
def evaluator_v4(pred, label):
    num_classes = pred.shape[1]
    task = 'binary' if num_classes == 2 else 'multiclass'  # 根据类别数量确定任务类型

    # 创建准确率指标对象（整体准确率）
    accuracy_metric = torchmetrics.Accuracy(average='macro', task=task, num_classes=num_classes)
    # 创建精确率指标对象（宏平均）
    precision_metric_macro = torchmetrics.Precision(average='macro', task=task, num_classes=num_classes)
    # 创建召回率指标对象（宏平均）
    recall_metric_macro = torchmetrics.Recall(average='macro', task=task, num_classes=num_classes)
    # 创建F1 - score指标对象（宏平均）
    f1_metric_macro = torchmetrics.F1Score(average='macro', task=task, num_classes=num_classes)

    pred_label = pred.argmax(dim=1)

    accuracy = accuracy_metric(pred.argmax(dim=1), label)
    precision_macro = precision_metric_macro(pred.argmax(dim=1), label)
    recall_macro = recall_metric_macro(pred.argmax(dim=1), label)
    f1_macro = f1_metric_macro(pred.argmax(dim=1), label)


    # accuracy = accuracy_score(label, pred_label)
    # precision = precision_score(label, pred_label)
    # f1 = f1_score(label, pred_label)
    # results['Accuracy'] = accuracy
    # results['Precision'] = precision
    # results['F1'] = f1
    tn, fp, fn, tp = confusion_matrix(label, pred_label).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # results["Sensitivity"] = sensitivity
    # results["Specificity"] = specificity

    # 计算每个类别的指标
    per_class_accuracy = [specificity,sensitivity]

    # for class_idx in range(num_classes):
    #     # 将当前类视为正类，其他类视为负类，进行二分类计算
    #     binary_pred = (pred.argmax(dim=1) == class_idx).int()
    #     binary_label = (label == class_idx).int()
    #
    #     # 使用二分类指标进行测量
    #     class_accuracy_metric = torchmetrics.Accuracy(task='binary')

    # per_class_accuracy.append(class_accuracy_metric(binary_pred, binary_label).item())


    return {
        'Accuracy': accuracy.item(),
        'Precision': precision_macro.item(),
        'Recall': recall_macro.item(),
        'F1': f1_macro.item(),
        'PerClassAccuracy': per_class_accuracy,
        'pred':pred_label,
        'label':label
    }
def evaluator_v3(pred, label):

    pred = pred.cpu().numpy()
    pred = expit(pred)
    label = label.cpu().numpy()
    results = {}
    auc = roc_auc_score(label, pred)
    results['AUC'] = auc
    pred_binary = (pred >= 0.5).astype(int)
    accuracy = accuracy_score(label, pred_binary)
    precision = precision_score(label, pred_binary)
    f1 = f1_score(label, pred_binary)
    results['Accuracy'] = accuracy
    results['Precision'] = precision
    results['F1'] = f1
    tn, fp, fn, tp = confusion_matrix(label, pred_binary).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    results["Sensitivity"] = sensitivity
    results["Specificity"] = specificity

    return results
def evaluator_v2(pred,orig):
    # Convert inputs to numpy arrays if they are not already
    pred = np.array(pred)
    orig = np.array(orig)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(orig, pred)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(orig, pred)

    # Calculate Coefficient of Determination (R²)
    r2 = r2_score(orig, pred)

    mape = mean_absolute_percentage_error(orig, pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE':mape
    }

def evaluator(pos,neg):
    results ={}

    pos = pos.squeeze(-1)
    neg = neg.squeeze(-1)
    all_scores = torch.cat([pos, neg]).cpu().numpy()
    all_scores = expit(all_scores)
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).cpu().numpy()
    auc = roc_auc_score(labels, all_scores)
    results["AUC"] = auc

    threshold = 0.5  # 假设分类阈值为 0.5
    predictions = (all_scores >= threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    # 计算 Precision, Recall 和 F1-score
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    results["Accuracy"] = accuracy
    results["Precision"] = precision
    results["F1"] = f1

    # 计算敏感性（Sensitivity）和特异性（Specificity）
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    results["Sensitivity"] = sensitivity
    results["Specificity"] = specificity

    return results