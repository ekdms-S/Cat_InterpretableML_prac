import os
from galvani import BioLogic
import pandas as pd
import numpy as np

from data.data_utils import get_exclude_idx
from utils import to_path


class RawLSVLoader_AgNi:
    def load_32V(self, n_trial, n_time=1042, n_loop=70, exclude_data=None):
        if self.catalyst == 'Ag':
            first_trial = 0 # Ag based data start with TRIAL 1
        else:
            first_trial = 1 # Ni based data start with TRIAL 2

        for i in range(n_trial):
            try:
                CO2 = BioLogic.MPRfile(self.CO2_dir.format(i + 1))
                N2 = BioLogic.MPRfile(self.N2_dir.format(i + 1))
                CO2_df = pd.DataFrame(CO2.data)
                N2_df = pd.DataFrame(N2.data)
                for j in range(n_loop):
                    CO2_i_df = CO2_df.iloc[(n_time * j + 2):n_time * (j + 1)][['<I>/mA']]
                    CO2_v_df = CO2_df.iloc[(n_time * j + 2):n_time * (j + 1)][['Ewe/V']]
                    N2_i_df = N2_df.loc[2:, ['<I>/mA']]
                    N2_v_df = N2_df.loc[2:, ['Ewe/V']]
                    if i == first_trial and j == 0:
                        CO2_i, CO2_v = CO2_i_df.to_numpy(), CO2_v_df.to_numpy()
                        N2_i, N2_v = N2_i_df.to_numpy(), N2_v_df.to_numpy()
                    else:
                        CO2_i = np.concatenate((CO2_i, CO2_i_df.to_numpy()), axis=1)
                        CO2_v = np.concatenate((CO2_v, CO2_v_df.to_numpy()), axis=1)
                        N2_i = np.concatenate((N2_i, N2_i_df.to_numpy()), axis=1)
                        N2_v = np.concatenate((N2_v, N2_v_df.to_numpy()), axis=1)
            except FileNotFoundError:
                pass

        exclude_idx = get_exclude_idx(CO2_i, exclude_data)

        CO2_i = np.delete(CO2_i, exclude_idx, axis=1)
        CO2_v = np.delete(CO2_v, exclude_idx, axis=1)
        N2_i = np.delete(N2_i, exclude_idx, axis=1)
        N2_v = np.delete(N2_v, exclude_idx, axis=1)

        return CO2_i, CO2_v, N2_i, N2_v

    def load_34V(self, n_trial, n_time=1107, n_loop=70, exclude_data=None, **kwargs):
        """
        part of Ag_based 3.4V data have different number of data points(1702) due to mechanical error of potentiostat.
        therefore, we should import them(N2_LSV TRAIL 1 ~ 4) separately
            -> N2_current_a, N2_potential_a / N2_current_b, N2_potential_b
        """
        if self.catalyst == 'Ag':
            for i in range(n_trial):
                try:
                    CO2 = BioLogic.MPRfile(self.CO2_dir.format(i + 1))
                    N2 = BioLogic.MPRfile(self.N2_dir.format(i + 1))
                    CO2_df = pd.DataFrame(CO2.data)
                    N2_df = pd.DataFrame(N2.data)
                    for j in range(n_loop):
                        CO2_i_df = CO2_df.iloc[(n_time * j + 2):n_time * (j + 1)][['<I>/mA']]
                        CO2_v_df = CO2_df.iloc[(n_time * j + 2):n_time * (j + 1)][['Ewe/V']]
                        N2_i_df = N2_df.loc[2:, ['<I>/mA']]
                        N2_v_df = N2_df.loc[2:, ['Ewe/V']]
                        if i == 0 and j == 0:
                            CO2_i, CO2_v = CO2_i_df.to_numpy(), CO2_v_df.to_numpy()
                            N2_i_a, N2_v_a = N2_i_df.to_numpy(), N2_v_df.to_numpy()
                        elif i == 4 and j == 0:
                            CO2_i = np.concatenate((CO2_i, CO2_i_df.to_numpy()), axis=1)
                            CO2_v = np.concatenate((CO2_v, CO2_v_df.to_numpy()), axis=1)
                            N2_i_b, N2_v_b = N2_i_df.to_numpy(), N2_v_df.to_numpy()
                        else:
                            CO2_i = np.concatenate((CO2_i, CO2_i_df.to_numpy()), axis=1)
                            CO2_v = np.concatenate((CO2_v, CO2_v_df.to_numpy()), axis=1)
                            if i < 4:
                                N2_i_a = np.concatenate((N2_i_a, N2_i_df.to_numpy()), axis=1)
                                N2_v_a = np.concatenate((N2_v_a, N2_v_df.to_numpy()), axis=1)
                            else:
                                N2_i_b = np.concatenate((N2_i_b, N2_i_df.to_numpy()), axis=1)
                                N2_v_b = np.concatenate((N2_v_b, N2_v_df.to_numpy()), axis=1)
                except FileNotFoundError:
                    pass

            CO2_exclude_idx = get_exclude_idx(CO2_i, kwargs['CO2_exclude_data'])
            N2_exclude_idx_a = get_exclude_idx(N2_i_a, kwargs['N2_exclude_data_a'])
            N2_exclude_idx_b = get_exclude_idx(N2_i_b, kwargs['N2_exclude_data_b'])

            CO2_i = np.delete(CO2_i, CO2_exclude_idx, axis=1)
            CO2_v = np.delete(CO2_v, CO2_exclude_idx, axis=1)
            N2_i_a = np.delete(N2_i_a, N2_exclude_idx_a, axis=1)
            N2_i_b = np.delete(N2_i_b, N2_exclude_idx_b, axis=1)
            N2_v_a = np.delete(N2_v_a, N2_exclude_idx_a, axis=1)
            N2_v_b = np.delete(N2_v_b, N2_exclude_idx_b, axis=1)

            return CO2_i, CO2_v, N2_i_a, N2_i_b, N2_v_a, N2_v_b

        else:
            for i in range(n_trial):
                try:
                    CO2 = BioLogic.MPRfile(self.CO2_dir.format(i + 1))
                    N2 = BioLogic.MPRfile(self.N2_dir.format(i + 1))
                    CO2_df = pd.DataFrame(CO2.data)
                    N2_df = pd.DataFrame(N2.data)
                    for j in range(n_loop):
                        CO2_i_df = CO2_df.iloc[(n_time * j + 2):n_time * (j + 1)][['<I>/mA']]
                        CO2_v_df = CO2_df.iloc[(n_time * j + 2):n_time * (j + 1)][['Ewe/V']]
                        N2_i_df = N2_df.loc[2:, ['<I>/mA']]
                        N2_v_df = N2_df.loc[2:, ['Ewe/V']]
                        if i == 0 and j == 0:
                            CO2_i, CO2_v = CO2_i_df.to_numpy(), CO2_v_df.to_numpy()
                            N2_i, N2_v = N2_i_df.to_numpy(), N2_v_df.to_numpy()
                        else:
                            CO2_i = np.concatenate((CO2_i, CO2_i_df.to_numpy()), axis=1)
                            CO2_v = np.concatenate((CO2_v, CO2_v_df.to_numpy()), axis=1)
                            N2_i = np.concatenate((N2_i, N2_i_df.to_numpy()), axis=1)
                            N2_v = np.concatenate((N2_v, N2_v_df.to_numpy()), axis=1)
                except FileNotFoundError:
                    pass

            exclude_idx = get_exclude_idx(CO2_i, exclude_data)

            CO2_i = np.delete(CO2_i, exclude_idx, axis=1)
            CO2_v = np.delete(CO2_v, exclude_idx, axis=1)
            N2_i = np.delete(N2_i, exclude_idx, axis=1)
            N2_v = np.delete(N2_v, exclude_idx, axis=1)

            return CO2_i, CO2_v, N2_i, N2_v

    def load_30V(self, n_trial, n_time=1107, n_loop=70, exclude_data=None):
        if self.catalyst == 'Ag':
            first_trial = 2 # Ag based data start with TRIAL3
        else:
            first_trial = 0 # Ni based data start with TRIAL1

        for i in range(n_trial):
            try:
                CO2 = BioLogic.MPRfile(self.CO2_dir.format(i + 1))
                N2 = BioLogic.MPRfile(self.N2_dir.format(i + 1))
                CO2_df = pd.DataFrame(CO2.data)
                N2_df = pd.DataFrame(N2.data)

                # last Ni AEM 3.0V LSV data(i.e. trial4 loop70) has only 9 data points
                #   -> exclude last loop.
                if self.catalyst == 'Ni' and i == 4:
                    n_loop = 69

                for j in range(n_loop):
                    CO2_i_df = CO2_df.iloc[(n_time * j + 2):n_time * (j + 1) - 130][['<I>/mA']]
                    CO2_v_df = CO2_df.iloc[n_time * j + 2:n_time * (j + 1) - 130][['Ewe/V']]
                    N2_i_df = N2_df.loc[2:976, ['<I>/mA']]
                    N2_v_df = N2_df.loc[2:976, ['Ewe/V']]
                    if i == first_trial and j == 0:
                        CO2_i = CO2_i_df.to_numpy()
                        CO2_v = CO2_v_df.to_numpy()
                        N2_i = N2_i_df.to_numpy()
                        N2_v = N2_v_df.to_numpy()
                    else:
                        CO2_i = np.concatenate((CO2_i, CO2_i_df.to_numpy()), axis=1)
                        CO2_v = np.concatenate((CO2_v, CO2_v_df.to_numpy()), axis=1)
                        N2_i = np.concatenate((N2_i, N2_i_df.to_numpy()), axis=1)
                        N2_v = np.concatenate((N2_v, N2_v_df.to_numpy()), axis=1)
            except FileNotFoundError:
                pass

        exclude_idx = get_exclude_idx(CO2_i, exclude_data, catalyst=self.catalyst, voltage=self.voltage)

        CO2_i = np.delete(CO2_i, exclude_idx, axis=1)
        CO2_v = np.delete(CO2_v, exclude_idx, axis=1)
        N2_i = np.delete(N2_i, exclude_idx, axis=1)
        N2_v = np.delete(N2_v, exclude_idx, axis=1)

        return CO2_i, CO2_v, N2_i, N2_v

    def __call__(self,
                 catalyst,
                 voltage,
                 n_trial,
                 exclude_data=None,
                 **kwargs):

        self.catalyst = catalyst
        self.voltage = voltage

        Dir = to_path(f'files/raw/{catalyst}/{voltage}V')
        CO2_mpr = f'CO2_LSV/LOOP_{-self.voltage}V_' + 'TRIAL{}_03_LSV_C01.mpr'
        N2_mpr = f'N2_LSV/N2_{-self.voltage}V_' + 'TRIAL{}_C01.mpr'
        self.CO2_dir = os.path.join(Dir, CO2_mpr)
        self.N2_dir = os.path.join(Dir, N2_mpr)

        if self.voltage == -3.2:
            CO2_i, CO2_v, N2_i, N2_v = self.load_32V(
                n_trial=n_trial, exclude_data=exclude_data
            )
        elif self.voltage == -3.4:
            if self.catalyst == 'Ag':
                CO2_i, CO2_v, N2_i_a, N2_i_b, N2_v_a, N2_v_b = self.load_34V(
                    n_trial=n_trial,
                    CO2_exclude_data=kwargs['CO2_exclude_data'],
                    N2_exclude_data_a=kwargs['N2_exclude_data_a'],
                    N2_exclude_data_b=kwargs['N2_exclude_data_b']
                )
                N2_i, N2_v = [N2_i_a, N2_i_b], [N2_v_a, N2_v_b]
            else:
                CO2_i, CO2_v, N2_i, N2_v = self.load_34V(
                    n_trial=n_trial, exclude_data=exclude_data
                )
        elif self.voltage == -3.0:
            CO2_i, CO2_v, N2_i, N2_v = self.load_30V(
                n_trial=n_trial, exclude_data=exclude_data
            )
        else:
            raise Exception(f'{self.voltage}V condition has no experiment data')

        return CO2_i, CO2_v, N2_i, N2_v


def RawLSVLodaer_Zn(voltage):
    Dir = to_path(f'files/raw/Zn/{voltage}V')
    CO2_mpr = f'CO2_LSV/LOOP_{-voltage}V_TRIAL1_03_LSV_C01.mpr'
    Ar_mpr = f'Ar_LSV/Ar_{-voltage}V_TRIAL1_C01.mpr'
    CO2_dir = os.path.join(Dir, CO2_mpr)
    Ar_dir = os.path.join(Dir, Ar_mpr)

    CO2 = BioLogic.MPRfile(CO2_dir)
    Ar = BioLogic.MPRfile(Ar_dir)
    CO2_df = pd.DataFrame(CO2.data)
    Ar_df = pd.DataFrame(Ar.data)

    n_time, n_loop = 1107, 70
    for j in range(n_loop):
        CO2_i_df = CO2_df.iloc[(n_time * j + 2):n_time * (j + 1)][['<I>/mA']]
        CO2_v_df = CO2_df.iloc[(n_time * j + 2):n_time * (j + 1)][['Ewe/V']]
        Ar_i_df = Ar_df.loc[2:, ['<I>/mA']]
        Ar_v_df = Ar_df.loc[2:, ['Ewe/V']]
        if j == 0:
            CO2_i, CO2_v = CO2_i_df.to_numpy(), CO2_v_df.to_numpy()
            Ar_i, Ar_v = Ar_i_df.to_numpy(), Ar_v_df.to_numpy()
        else:
            CO2_i = np.concatenate((CO2_i, CO2_i_df.to_numpy()), axis=1)
            CO2_v = np.concatenate((CO2_v, CO2_v_df.to_numpy()), axis=1)
            Ar_i = np.concatenate((Ar_i, Ar_i_df.to_numpy()), axis=1)
            Ar_v = np.concatenate((Ar_v, Ar_v_df.to_numpy()), axis=1)

    return CO2_i, CO2_v, Ar_i, Ar_v
