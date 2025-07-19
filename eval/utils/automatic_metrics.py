import evaluate
import numpy as np


class MetricsComputer:
    def __init__(self, prediction_list, gt_list):
        self.pred_list = prediction_list
        self.gt_list = gt_list

    def compute_BLEU(self):
        bleu = evaluate.load("bleu")
        results = bleu.compute(predictions=self.pred_list, references=self.gt_list)
        return results["bleu"]
    
    def compute_ROUGE(self):
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=self.pred_list, references=self.gt_list)
        return results
    
    def compute_BERTScore(self):
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(predictions=self.pred_list, references=self.gt_list, lang="en")
        return  np.mean(results['f1']) # based on this paper, just mean of f1: https://github.com/StanfordMIMI/clin-summ/blob/main/src/calc_metrics.py
    
    def compute_METEOR(self):
        meteor= evaluate.load('meteor')
        results = meteor.compute(predictions=self.pred_list, references=self.gt_list)
        return results["meteor"]
    
    



