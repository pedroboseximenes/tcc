from codecarbon import EmissionsTracker
import pandas as pd

# ========================================================================================
# LOGGER CONFIG
# ========================================================================================
from utils.ModeloBase import ModeloBase
from utils.logger import Logger
import os
import utils.utils as util
import torch
import torch.utils.data as data
import utils.plotUtils as plot
import torch.nn.functional as F
import time
from utils.lstmModel import LstmModel


class LSTM(ModeloBase):
    def run(self, index):
        self.logger.info("=" * 90)
        self.logger.info(f"Iniciando script LSTM (PyTorch) {self.base_dados} com suporte a GPU e logs detalhados.")
        self.logger.info("="*90)
        self.logger.info(
            f"Iniciando experimento: lookback={self.lookback}, "
            f"hidden_dim={self.config['hidden_dim']}, layers={self.config['layer_dim']}, "
            f"lr={self.config['learning_rate']}, drop={self.config['drop_rate']}"
        )

        # ---------- FASE: criar sequências para esse lookback ----------
        X, y = util.create_sequence(self.ts_scaled_df.values, self.lookback)
        X_train, X_test = util.split_last_n(X, n_test=self.num_test)
        y_train, y_test = util.split_last_n(y, n_test=self.num_test)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.config['device'])
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.config['device'])
        X_test  = torch.tensor(X_test,  dtype=torch.float32).to(self.config['device'])
        y_test  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1).to(self.config['device'])

        self.logger.info(f"Shape treino: {X_train.shape}, teste: {X_test.shape}")
        # ---------- FASE: modelo ----------
        model = LstmModel(
            input_dim=X_train.shape[2],
            hidden_dim=self.config['hidden_dim'],
            layer_dim=self.config['layer_dim'],
            output_dim=1,
            drop_rate=self.config['drop_rate']
        ).to(self.config['device'])

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])

        train_loader = data.DataLoader(
            data.TensorDataset(X_train, y_train),
            shuffle=True,
            batch_size=self.batch_size
        )

        self.logger.info(f"Treinando por {self.epocas} épocas...")
        inicio = time.time()

        for epoch in range(1, self.epocas + 1):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)

                mse = F.mse_loss(outputs, y_batch, reduction='mean')
                mae = F.l1_loss(outputs, y_batch, reduction='mean')

                loss = (1.0 * (mse + mae))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 50 == 0 or epoch == 1:
                self.logger.info(
                    f"[EXP] Época {epoch}/{self.epocas} - Loss: {epoch_loss / len(train_loader):.6f}"
                )
        tempoTreinamento = (time.time() - inicio)/60 
        self.logger.info(f"Treinamento concluído em {tempoTreinamento:.2f} minutos")

        # ---------- FASE: avaliação ----------
        train_pred = util.predict_in_batches(model, X_train, self.config['device'], batch_size=self.batch_size)
        y_pred  = util.predict_in_batches(model, X_test,  self.config['device'], batch_size=self.batch_size)

        # desescalar chuva
        self.logger.info(f"Calculando erro para o train")
        teste_size = len(self.timeseries) - len(train_pred) - self.lookback
        y_pred_mm_train, y_true_mm_train = util.desescalar_pred_generico(
            train_pred,
            self.colunas_normalizar,
            scaler=self.scaler,
            ts_scaled=self.ts_scaled_df,
            timeseries=self.timeseries,
            target='chuva',
            start=teste_size,
            lookback=self.lookback
        )
        #y_pred_mm_train, y_true_mm_train = train_pred.squeeze(-1).detach().cpu().numpy(),  y_train.squeeze(-1).detach().cpu().numpy()
        rmseTrain, mseTrain , maeTrain, csiTrain = util.calcular_erros(logger=self.logger, dadoPrevisao=y_pred_mm_train, dadoReal=y_true_mm_train)
        self.logger.info(f"train_pred TRAIN mm min/max: {float(y_pred_mm_train.min())}, {float(y_pred_mm_train.max())}")
        self.logger.info(f"train_TRUE TRAIN mm min/max: {float(y_true_mm_train.min())}, {float(y_true_mm_train.max())}")

        self.logger.info(" Gerando gráficos TRAIN...")
        plot.gerar_plot_dois_eixo(eixo_x=y_true_mm_train, eixo_y=y_pred_mm_train, titulo=f"TRAIN [{index}] - lstm{self.base_dados}_lookback={self.lookback}_neuronios={self.config['hidden_dim']}_camada={self.config['layer_dim']}_lr={self.config['learning_rate']}_droprate={self.config['drop_rate']}", xlabel="Amostra", ylabel="Chuva", legenda=['Real', 'Previsto'], dataset=self.base_dados,index=index)
        self.logger.info(" Gráficos gerados TRAIN ...")

        self.logger.info(f"Calculando erro para parte de teste")
        validation_size = len(self.timeseries) - len(y_pred) - self.lookback
        y_pred_mm, y_true_mm = util.desescalar_pred_generico(
            y_pred,
            self.colunas_normalizar,
            scaler=self.scaler,
            ts_scaled=self.ts_scaled_df,
            timeseries=self.timeseries,
            target='chuva',
            start=validation_size,
            lookback=self.lookback
        )
        #y_pred_mm, y_true_mm = y_pred.squeeze(-1).detach().cpu().numpy(),  y_test.squeeze(-1).detach().cpu().numpy()
        rmse, mse , mae, csi = util.calcular_erros(logger=self.logger, dadoPrevisao=y_pred_mm, dadoReal=y_true_mm)
        self.logger.info(f"y_pred mm min/max: {float(y_pred_mm.min())}, {float(y_pred_mm.max())}")
        self.logger.info(f"y_TRUE mm min/max: {float(y_true_mm.min())}, {float(y_true_mm.max())}")

        self.logger.info(" Gerando gráficos...")
        plot.gerar_plot_dois_eixo(eixo_x=y_true_mm, eixo_y=y_pred_mm, titulo=f"TEST [{index}] - lstm{self.base_dados}_lookback={self.lookback}_neuronios={self.config['hidden_dim']}_camada={self.config['layer_dim']}_lr={self.config['learning_rate']}_droprate={self.config['drop_rate']}", xlabel="Amostra", ylabel="Chuva", legenda=['Real', 'Previsto'],dataset=self.base_dados,index=index)
        self.logger.info(" Gráficos gerados...")
        self.logger.info("=" * 90)
        self.logger.info("Execução finalizada com sucesso.")
        self.logger.info(f"Dispositivo utilizado: {self.config['device']}")
        self.logger.info("=" * 90)
        

        return {
        "lookback": self.lookback,
        "hidden_dim": self.config['hidden_dim'],
        "layer_dim": self.config['layer_dim'],
        "learning_rate": self.config['learning_rate'],
        "drop_rate": self.config['drop_rate'],
        "rmseTrain": rmseTrain,
        "mseTrain": mseTrain,
        "maeTrain": maeTrain,
        "csiTrain": csiTrain,
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "csi": csi,
        "tempoTreinamento":tempoTreinamento,
        "y_pred": y_pred_mm,
    }
