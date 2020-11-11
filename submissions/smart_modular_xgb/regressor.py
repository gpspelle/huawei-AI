import numpy as np
from sklearn import multioutput
import xgboost as xgb

class Regressor():
    def _init_(self):
        super()._init_()
        self.model = None

    def fit(self, X, y):        
        # Create empty model made
        self.model_bag = dict()
        
        # Data bag
        self.data_bag = dict()
        
        max_cascade_size = 9
        for i in range(1, max_cascade_size):
            self.data_bag[i] = dict()
            self.model_bag[i] = dict()
        
        
        for inp, out in zip(X, y):
            metadata = inp[0]
            signal = inp[1]
            
            number_modules_in_cascade = len(metadata)
            metadata_str = ""
            for i in range(number_modules_in_cascade):
                metadata_str += (metadata[i][0] + "_" + str(metadata[i][1][0]) + "_" + str(metadata[i][1][1]) + "-")

            metadata_str = metadata_str[:-1]

            try:
                self.data_bag[number_modules_in_cascade][metadata_str].append([signal, out])
            except:
                self.data_bag[number_modules_in_cascade][metadata_str] = [[signal, out]]
                

        all_train_input_EDFA = []
        all_train_output_EDFA = []
        all_train_input_SMF = []
        all_train_output_SMF = []
        # Train one model per size 1 cascade
        for metadata_str in list(self.data_bag[1].keys()):
            # Train only with size 1 cascades
            train_data = np.asarray((self.data_bag[1][metadata_str]))
            train_input, train_output = train_data[:, 0], train_data[:, 1]

            if 'EDFA' in metadata_str:
                all_train_input_EDFA += list(train_input)
                all_train_output_EDFA += list(train_output)
            else:
                all_train_input_SMF += list(train_input)
                all_train_output_SMF += list(train_output)

            self.model_bag[1][metadata_str] = multioutput.MultiOutputRegressor(xgb.XGBRegressor()).fit(train_input, train_output)

        self.model_bag[1]['joker_EDFA'] = multioutput.MultiOutputRegressor(xgb.XGBRegressor()).fit(np.asarray(all_train_input_EDFA), np.asarray(all_train_output_EDFA))
        self.model_bag[1]['joker_SMF'] = multioutput.MultiOutputRegressor(xgb.XGBRegressor()).fit(np.asarray(all_train_input_SMF), np.asarray(all_train_output_SMF))

        # Now, let's train also with the size 2 cascades
        for metadata_str in list(self.data_bag[2].keys()):

            metadata_split_str = metadata_str.split('-')
            first_individual_module = metadata_split_str[0]
            second_individual_module = metadata_split_str[1]

            try:
                model = self.model_bag[1][first_individual_module]
            except:
                if 'EDFA' in first_individual_module:
                    model = self.model_bag[1]['joker_EDFA']
                else:
                    model = self.model_bag[1]['joker_SMF']

            data = np.asarray(self.data_bag[2][metadata_str])
            train_inp, train_out = data[:, 0], data[:, 1]

            pred = model.predict(train_inp)
            pred = pred * (pred > 0)

            if second_individual_module in self.model_bag[1]:
                self.model_bag[1][second_individual_module].fit(pred, train_out)
            else:
                if 'EDFA' in second_individual_module:
                    self.model_bag[1]['joker_EDFA'].fit(pred, train_out)
                else:
                    self.model_bag[1]['joker_SMF'].fit(pred, train_out)


    def predict(self, X):
        
        preds = []
        for inp in X:
            metadata = inp[0]
            signal = inp[1]
            
            for module in metadata:
                metadata_str = module[0] + "_" + str(module[1][0]) + "_" + str(module[1][1])

                try:
                    model = self.model_bag[1][metadata_str]
                except:
                    if 'EDFA' in metadata_str:
                        model = self.model_bag[1]['joker_EDFA']
                    else:
                        model = self.model_bag[1]['joker_SMF']

                pred = model.predict(np.asarray(signal).reshape(1, -1))
                pred = pred * (pred > 0)

                # Use previous pred as the new input
                signal = pred
            
            preds.append(pred[0])

        return np.asarray(preds)
