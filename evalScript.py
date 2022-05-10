import pandas as pd
import torch
import torchmetrics
import os
import yaml
from Utils import dataUtils, modelUtils
from tqdm import tqdm


# ? Eval from https://github.com/xieh97/dcase2022-audio-retrieval
# ? Functions go refractored to a class and adapted for the coursework
class model_stats:
    def __init__(self) -> None:
        self.gt_csv = "test.gt.csv"  # ground truth for Clotho evaluation data
        self.pred_csv = "test.output.csv"  # baseline system retrieved output for Clotho evaluation data

    def __load_clotho_csv(self, fpath):
        caption_fname = {}

        rows = pd.read_csv(fpath)
        rows = [list(row) for row in rows.values]

        for row in rows:
            for cap in row[1:]:  # captions
                caption_fname[cap] = row[0]

        return caption_fname

    def __load_output_csv(self, fpath):
        caption_fnames = {}

        rows = pd.read_csv(fpath)
        rows = [list(row) for row in rows.values]

        for row in rows:
            caption_fnames[row[0]] = row[1:]

        return caption_fnames

    def retrieval_metrics(self, gt_csv, pred_csv):
        # Initialize retrieval metrics
        R1 = torchmetrics.RetrievalRecall(
            empty_target_action="neg", compute_on_step=False, k=1
        )
        R5 = torchmetrics.RetrievalRecall(
            empty_target_action="neg", compute_on_step=False, k=5
        )
        R10 = torchmetrics.RetrievalRecall(
            empty_target_action="neg", compute_on_step=False, k=10
        )
        mAP10 = torchmetrics.RetrievalMAP(
            empty_target_action="neg", compute_on_step=False
        )

        gt_items = self.__load_clotho_csv(gt_csv)
        pred_items = self.__load_output_csv(pred_csv)

        for i, cap in enumerate(gt_items):
            gt_fname = gt_items[cap]
            pred_fnames = pred_items[cap]

            preds = torch.as_tensor(
                [1.0 / (pred_fnames.index(pred) + 1) for pred in pred_fnames],
                dtype=torch.float,
            )
            targets = torch.as_tensor(
                [gt_fname == pred for pred in pred_fnames], dtype=torch.bool
            )
            indexes = torch.as_tensor([i for pred in pred_fnames], dtype=torch.long)

            # Update retrieval metrics
            R1(preds, targets, indexes=indexes)
            R5(preds, targets, indexes=indexes)
            R10(preds, targets, indexes=indexes)
            mAP10(preds[:10], targets[:10], indexes=indexes[:10])

        metrics = {
            "R1": R1.compute().item(),  # 0.03
            "R5": R5.compute().item(),  # 0.11
            "R10": R10.compute().item(),  # 0.19
            "mAP10": mAP10.compute().item(),  # 0.07
        }

        for key in metrics:
            print(key, "{:.2f}".format(metrics[key]))


class checkpoint_eval:
    def __transform(self, model, dataset, index, device=None):
        audio, query, info = dataset[index]

        audio = torch.unsqueeze(audio, dim=0).to(device=device)
        query = torch.unsqueeze(query, dim=0).to(device=device)

        audio_emb, query_emb = model(audio, query, [query.size(-1)])

        audio_emb = torch.squeeze(audio_emb, dim=0).to(device=device)
        query_emb = torch.squeeze(query_emb, dim=0).to(device=device)

        return audio_emb, query_emb, info

    def __audio_retrieval(self, model, caption_dataset, K=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device=device)

        model.eval()

        with torch.no_grad():

            fid_embs, fid_fnames = {}, {}
            cid_embs, cid_infos = {}, {}

            # Encode audio signals and captions
            for cap_ind in tqdm(
                range(len(caption_dataset)), desc="Loading caption data"
            ):
                audio_emb, query_emb, info = self.__transform(
                    model, caption_dataset, cap_ind, device
                )

                fid_embs[info["fid"]] = audio_emb
                fid_fnames[info["fid"]] = info["fname"]

                cid_embs[info["cid"]] = query_emb
                cid_infos[info["cid"]] = info

            # Stack audio embeddings
            audio_embs, fnames = [], []
            for fid in tqdm(fid_embs, desc="Formating audio embeddings and file names"):
                audio_embs.append(fid_embs[fid])
                fnames.append(fid_fnames[fid])

            audio_embs = torch.vstack(audio_embs)  # dim [N, E]

            # Compute similarities
            output_rows = []
            for cid in tqdm(cid_embs, desc="Caption-File similarity"):
                # ? mm Performs a matrix multiplication of the matrices input and mat2.
                sims = (
                    torch.mm(torch.vstack([cid_embs[cid]]), audio_embs.T)
                    .flatten()
                    .to(device=device)
                )

                sorted_idx = torch.argsort(sims, dim=-1, descending=True)

                csv_row = [cid_infos[cid]["caption"]]  # caption
                for idx in sorted_idx[:K]:  # top-K retrieved fnames
                    csv_row.append(fnames[idx])

                output_rows.append(csv_row)

            return output_rows

    def eval_checkpoint(self, config, checkpoint_dir):
        # Load config
        training_config = config["training"]

        # Load evaluation
        caption_datasets, vocabulary = dataUtils.load_data(config["eval_data"])

        # Initialize a model instance
        model_config = config[training_config["model"]]
        model = modelUtils.get_model(model_config, vocabulary)
        print(model)

        # Restore model states
        model = modelUtils.restore(model, checkpoint_dir)
        model.eval()

        # Retrieve audio files for evaluation captions
        for split in ["evaluation"]:
            output = self.__audio_retrieval(model, caption_datasets[split], K=10)

            csv_fields = [
                "caption",
                "file_name_1",
                "file_name_2",
                "file_name_3",
                "file_name_4",
                "file_name_5",
                "file_name_6",
                "file_name_7",
                "file_name_8",
                "file_name_9",
                "file_name_10",
            ]

            output = pd.DataFrame(data=output, columns=csv_fields)
            output.to_csv(
                os.path.join(checkpoint_dir, "{}.output.csv".format(split)), index=False
            )
            print("Saved", "{}.output.csv".format(split))


if __name__ == "__main__":
    with open("conf.yaml", "rb") as stream:
        conf = yaml.full_load(stream)

    #checkpointEval = checkpoint_eval()

    #checkpointEval.eval_checkpoint(conf, "model_results/CNN14/checkpoint")
    #modelStats = model_stats()
    #modelStats.retrieval_metrics('test.gt.csv','model_results/CNN14/checkpoint/evaluation.output.csv')


# retrieval_metrics(gt_csv, pred_csv)
