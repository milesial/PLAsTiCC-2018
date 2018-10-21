""" Attempt to predict hostgal_specz from less precise hostgal_photoz """

from torch import Tensor
from torch import nn
from torch import optim
from torch.utils import data
from ignite import engine
from ignite import metrics
from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar
import numpy as np

from utils import data_utils
from utils.ignite_utils import Identity, Average


useful_columns = [
    'ddf',
    'distmod',
    'mwebv',
    'hostgal_photoz',
    'hostgal_photoz_err',
    'hostgal_specz',
]


class RedshiftDataset(data.Dataset):
    def __init__(self, type='train'):
        if type == 'train':
            self.data = data_utils.get_train_set_metadata()
        elif type == 'test':
            self.data = data_utils.get_test_set_metadata()
        else:
            raise ValueError('Type must be test or train')

        self.data = self.data[useful_columns]
        self.data.dropna(inplace=True)
        self.data = self.data[self.data['hostgal_specz'] != 0]
        self.data = self.data[self.data['hostgal_photoz'] != 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Sometimes (after calling data.random_split), item is a tensor,
        # and pandas does not like it. See https://github.com/pytorch/pytorch/issues/10165
        if type(item) == Tensor:
            item = item.item()
        row = self.data.iloc[item, :]
        # Returns (features, target)
        return Tensor(row.drop('hostgal_specz').values.astype(np.float32)), \
               Tensor([row['hostgal_specz']])


def get_model(n_features):
    model = nn.Sequential(
        nn.Linear(n_features, 20),
        nn.Linear(20, 20),
        nn.LeakyReLU(),
        nn.Linear(20, 1),
    )
    return model


def get_data_loaders(batch_size, val_split):
    # Here, we use the kaggle test set to add data
    # It does not have the targets class, but some rows have hostgal_specz: we use those
    # This represents a lot more data than in the train set
    full_dataset = data.ConcatDataset([RedshiftDataset('train'), RedshiftDataset('test')])
    full_length = len(full_dataset)
    val_length = int(val_split * full_length)
    [train, val] = data.random_split(full_dataset, [full_length - val_length, val_length])
    train_loader = data.DataLoader(dataset=train,
                                   batch_size=batch_size,
                                   shuffle=True)
    val_loader = data.DataLoader(dataset=val,
                                 batch_size=batch_size)
    return train_loader, val_loader


def get_engines(learning_rate=0.01):
    # We remove 1 because of the target column
    model = get_model(n_features=len(useful_columns) - 1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    trainer = engine.create_supervised_trainer(model, optimizer, loss)
    evaluator = engine.create_supervised_evaluator(model, metrics={
        'mae': metrics.MeanAbsoluteError(),
        'loss': metrics.Loss(loss),
        'avg': metrics.RunningAverage(metrics.Loss(loss))
    })

    # workaround for using the ProgressBar with training loss
    Identity().attach(trainer, 'loss')
    Average().attach(trainer, 'avg_loss')
    return trainer, evaluator


def assign_event_handlers(trainer, evaluator, val_set):
    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        print("\nTraining Results - Epoch: {} : Avg loss: {:.3f}"
              .format(trainer.state.epoch, trainer.state.metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_set)
        metrics_eval = evaluator.state.metrics
        print("Validation Results - Epoch: {} Avg loss: {:.3f}, Avg abs. error: {:.2f}"
              .format(trainer.state.epoch, metrics_eval['loss'], metrics_eval['mae']))


if __name__ == '__main__':
    train_set, val_set = get_data_loaders(batch_size=128, val_split=0.3)

    train, eval = get_engines(learning_rate=0.02)
    assign_event_handlers(train, eval, val_set)

    train.run(train_set, max_epochs=30)

