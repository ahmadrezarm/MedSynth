import evaluate
import numpy as np
from medcon import UMLSScorer 

#TODO: Add PROMETHEUS
class Metrics:
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

    def compute_MEDCON(self):
        medcon = UMLSScorer()
        score_medcon = medcon(self.gt_list, self.pred_list)
        return score_medcon




prediction_list = ["hello there general kenobi", "i am the senate", "do or do not, there is no try"]
gt_list = ["hello there general kenobi", "no, i am the father", "do or do not, there is no try"]

metrics = Metrics(prediction_list, gt_list)
average_bleu_score = metrics.compute_BLEU()
print(f" BLEU Score: {average_bleu_score}")
print(f"ROUGE 1 is: {metrics.compute_ROUGE()['rouge1']}")
print(f"ROUGE 2 is: {metrics.compute_ROUGE()['rouge2']}")
print(f"ROUGE L is: {metrics.compute_ROUGE()['rougeL']}")
print(f"ROUGE LSum is: {metrics.compute_ROUGE()['rougeLsum']}")
print(f"BERTScore is: {metrics.compute_BERTScore()}")
#print(f"MEDCON score is: {metrics.compute_MEDCON()}") : awaiting UML licence to install QuickUMLS
