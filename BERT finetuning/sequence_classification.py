import itertools
from torch.optim.lr_scheduler import StepLR
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from read_data import *
import numpy as np
# from torchcrf import CRF
from tqdm import trange

# Load the BertForSequenceClassification model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 5,
    output_attentions = False,
    output_hidden_states = False,
)




# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(),
                              lr = 5e-5,
                              eps = 1e-08
                              )
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Run on GPU
model.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
epochs = 20

for _ in trange(epochs, desc='Epoch'):

    # ========== Training ==========

    # Set model to training mode
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        # Forward pass
        train_output = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        # Backward pass
        train_output.loss.backward()
        optimizer.step()
        # Update tracking variables
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1


    # ========== Validation ==========

    # Set model to evaluation mode
    model.eval()

    predictions=[]
    lbls=[]
    # Tracking variables
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []
    reports=[]

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # Forward pass
            eval_output = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)
        logits = eval_output.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(np.argmax(logits, axis = 1).flatten())
        print('------------predictions-------')
        print(predictions)
        print('lbsl')
        lbls.append(label_ids)
        print(lbls)
        # Calculate validation metrics
        # b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)


        # val_accuracy.append(b_accuracy)
        # # Update precision only when (tp + fp) !=0; ignore nan
        # if b_precision != 'nan': val_precision.append(b_precision)
        # # Update recall only when (tp + fn) !=0; ignore nan
        # if b_recall != 'nan': val_recall.append(b_recall)
        # # Update specificity only when (tn + fp) !=0; ignore nan
        # if b_specificity != 'nan': val_specificity.append(b_specificity)
    scheduler.step()

    # _, _, _, _, report = b_metrics(predictions, lbls)
    print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    # print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy) / len(val_accuracy)))
    # print('\t - Validation Precision: {:.4f}'.format(sum(val_precision) / len(val_precision)) if len(val_precision) > 0 else '\t - Validation Precision: NaN')
    # print('\t - Validation Recall: {:.4f}'.format(sum(val_recall) / len(val_recall)) if len(
    #     val_recall) > 0 else '\t - Validation Recall: NaN')
    # print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity) / len(val_specificity)) if len(
    #     val_specificity) > 0 else '\t - Validation Specificity: NaN')

    prediction = list(itertools.chain(*predictions))

    groud_truth=list(itertools.chain(*lbls))
    report=classification_report(groud_truth,prediction,target_names=['0', 'cause', 'enable', 'intend', 'prevent'])
    print(report)


PATH = "entire_model_sequence_joined_data.pt"

# Save
torch.save(model, PATH)