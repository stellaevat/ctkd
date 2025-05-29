import evaluate
import torch
from tqdm import tqdm
from pprint import pprint
from addict import Dict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

from dskd_distiller import DSKD
from dskd_dataset import build_student_teacher_dataset

def show_results(epoch, loss, ce_loss, kd_loss, metrics):
  print(f"\n\nEPOCH {epoch}\n")
  print(f"Training loss: {loss}")
  print(f"CE loss: {ce_loss}")
  print(f"KD loss: {kd_loss}")
  pprint(metrics)
  print()


def full_determinism(seed=74):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def eval_student(distiller, dataset, device, batch_size=8, testing=False, verbose=False):
    split = "test" if testing else "dev"
    dataloader = DataLoader(dataset[split], batch_size=batch_size)
    accuracy = evaluate.load('accuracy')

    distiller.eval()
    avg_loss = 0.0
    for batch in dataloader:
        batch = {k : v.to(device) for (k, v) in batch.items()}
        batch_ratio = len(batch) / len(dataset[split])

        with torch.no_grad():
            loss, _, _, logits = distiller(**batch)

        avg_loss += loss * batch_ratio

        predictions = torch.argmax(logits.cpu(), dim=-1)
        accuracy.add_batch(predictions=predictions, references=batch["s_label"].cpu())

    metric_dict = {("Test loss" if testing else "Validation loss") : avg_loss}
    metric_dict.update(accuracy.compute())

    # TODO: add more metrics (e.g. Rouge-L)

    return metric_dict


def train_student(distiller, device, data_dir, data_splits=["train", "dev"], epochs=10, batch_size=4, learning_rate=2e-4, verbose=True):
    dataset = build_student_teacher_dataset(distiller, data_dir, data_splits)
    dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=batch_size)
    num_training_steps = epochs * len(dataloader)

    optimizer = AdamW(distiller.parameters(), lr=learning_rate)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    progress_bar = tqdm(range(num_training_steps), position=0, leave=True)

    distiller.train()
    for epoch in range(1, epochs+1):
        avg_loss, avg_ce_loss, avg_kd_loss = 0.0, 0.0, 0.0
        for batch in dataloader:
            batch = {k : v.to(device) for (k, v) in batch.items()}
            batch_ratio = len(batch) / len(dataset["train"])

            loss, ce_loss, kd_loss, _ = distiller(batch)
            
            avg_loss += loss * batch_ratio
            avg_ce_loss += ce_loss * batch_ratio
            avg_loss += kd_loss * batch_ratio

            distiller.backward(loss)
            distiller.step()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


        metrics = eval_student(distiller, dataset, device, batch_size=batch_size)
        if verbose:
            show_results(avg_loss, avg_ce_loss, avg_kd_loss, metrics)
            
if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_dir = "data/dolly"

    args = Dict({
        "s_path" : "gpt2",
        "s_type" : "gpt2",
        "s_dtype" : torch.bfloat16,
        "t_path" : "qwen/Qwen1.5-1.8B",
        "t_type" : "qwen",
        "t_dtype" : torch.bfloat16,
        "proj_path" : "models/projectors.pth",
        "kl_temperature" : 1,
        "kd_weight" : 0.5,
    })
    distiller = DSKD(args, device)

    full_determinism()
    train_student(distiller, device, data_dir)

