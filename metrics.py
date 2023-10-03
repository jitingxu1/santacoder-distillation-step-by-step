import numpy as np
import wandb
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

# TODO: Need to find a good way to measure accuracy of codes. 
def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        table_data = []
        bleu_scores = []
        accs = []

        for pred, label in zip(predictions, labels):

          predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
          decoded_preds = tokenizer.batch_decode(
                  predictions,
                  skip_special_tokens=True
          )
          labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
          decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

          
          ref_bleu = []
          gen_bleu = []
        
          for l, r in zip(decoded_preds, decoded_labels):
              gen_bleu.append(l.split())
              ref_bleu.append([r.split()])
              table_data.append([l, r])

          cc = SmoothingFunction()
          score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
          acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
          accs.append(acc)
          bleu_scores.append(score_bleu)

        
        table = wandb.Table(data=table_data, columns=["Predictions", "Labels"])

        wandb.log({
            "table": table,
            })
        
        return {
          'accuracy_code': acc[0],
          'accuracy_rationale': acc[1],
          'bleu_code': bleu_scores[0],
          'bleu_rationale': bleu_scores[0]
        }



    return compute_metrics