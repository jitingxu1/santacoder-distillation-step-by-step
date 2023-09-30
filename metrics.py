import numpy as np
import wandb

def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Too many empty outputs from T5
        no_empty_output = sum([1 for p in predictions if len(p) < 5 ])
        wandb.log({
            "num_empty_t5_output": no_empty_output,
            })
        total_acc = 0.0
        data = []
        n = len(predictions)
        for p, l in zip(predictions, labels):
            p = np.where(
                p != -100,
                p,
                tokenizer.pad_token_id
            )
            decoded_preds = tokenizer.batch_decode(
                p,
                max_length=512,
                skip_special_tokens=True
            )
            l = np.where(l != -100, l, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(
                l,
                skip_special_tokens=True
            )
            acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
            total_acc += acc
            data.append([decoded_preds, decoded_labels, acc])

        columns=["T5 output", "Santa coder output", "Accuracy"]
        example_table = wandb.Table(data=data, columns=columns)
        wandb.log({"Example": example_table})
        # acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': total_acc / n}

    return compute_metrics