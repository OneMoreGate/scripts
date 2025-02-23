from matplotlib import axes
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd 
import numpy as np 
import os
import shutil

class Measurments():
    # инициализация, проверка существования папки, проверка пустоты папки
    def __init__(self, sample: str) -> None:
        if not os.path.exists(sample):
            raise FileNotFoundError(f'directory \'{sample}\' is not exist')
        elif  (len(os.listdir(sample)) == 0):
            raise ValueError(f'directory \'{sample}\' is empty')
        else:
            self.__sample = sample
            self.__sample_path = os.path.abspath(sample)
            self.__create_dict_of_measurments()
            
    # поиск всех вложенных директорий в основной папке
    def __find_contacts_folder(self) -> None:
        self.__contacts_list = []
        for i in os.listdir(self.__sample):
            if os.path.isdir(os.path.join(self.__sample, i)):
                self.__contacts_list.append(f'{i}')

    # проверка, что файл типа data
    def __is_data_file(self, path: str) -> bool:
        if path.split('.')[-1] == 'data':
            return True
        else:
            return False
        
    # создание пути из названия папок
    def __create_path(self, path_list: list[str]) -> str:
        return os.path.join(self.__sample_path, *path_list)

    # считывание данных об измерениях из файлов во вложенных директориях
    def __create_dict_of_measurments(self) -> None:
        self.__find_contacts_folder()
        if len(self.__contacts_list) == 0:
            raise ValueError(f'directory \'{self.__sample}\' does not contain subdirectories')
        self.__dict_of_measurments = {}
        for contact in self.__contacts_list:
            subdir_catalogs_list = [file for file in os.listdir(self.__create_path([contact])) if self.__is_data_file(self.__create_path([contact, file]))]
            if len(subdir_catalogs_list) == 0:
                continue
            contact_measurs_type = {}
            for measure in subdir_catalogs_list:
                file_path = self.__create_path([contact, measure])
                with open(file_path) as file:
                    measur_type = '_'.join(file.readlines()[1].split()[1:3])
                contact_measurs_type[measure.replace('.data', '')] = measur_type
            sorted_contact_measurs_type = dict(sorted(contact_measurs_type.items(), key=lambda item: int(item[0])))
            self.__dict_of_measurments[contact] = sorted_contact_measurs_type
        
    # возврат полного словаря со всеми контактами и измерениямими 
    def get_full_dict(self)-> dict:
        return self.__dict_of_measurments
        
    # абсолютный путь к указанной папке
    def get_abspath(self) -> str:
        return self.__sample_path
    
    def get_contacts(self) -> list:
        return list(self.__contacts_list)
    
    # получение словаря с измерениями только с одного контакта
    def get_contact_dict(self, contact_name: str | int) -> dict:
        if not isinstance(contact_name, str | int):
            raise ValueError(f'contact_name must be str or int type')
        elif str(contact_name) not in list(self.__dict_of_measurments.keys()):
            raise ValueError(f'{contact_name} not exist in {self.__sample}')
        else:
            return {str(contact_name) :self.__dict_of_measurments[str(contact_name)]}
    
    # удаление контакта или контактов из словаря с измерениями
    def delete_contacts(self, contacts_name: str | list) -> None:
        if not isinstance(contacts_name, list):
            contacts_name = [contacts_name]
        for contact in contacts_name:
            try:
                self.__contacts_list.remove(str(contact))
                del self.__dict_of_measurments[str(contact)]
            except:
                continue
    
    # удаление измерений из контакта 
    def delete_measurments(self, del_dict: dict) -> None:
        for contact in list(del_dict.keys()):
            if str(contact) in self.__contacts_list:
                if not isinstance(del_dict[contact], list):
                    raise ValueError(f'dict values must be {list} type')
                for measur in del_dict[contact]:
                    if str(measur) in list(self.__dict_of_measurments[str(contact)]):
                        self.__dict_of_measurments[str(contact)].pop(str(measur), None)
                    else:
                        continue
            else:
                continue

# возвращает датафрейм с ВАХ конкретного измерения
def get_DC_IV_data(data_path: str) -> pd.DataFrame:
    DC_IV_dataframe = pd.read_csv(data_path, delimiter='   ', skiprows=16, engine='python', header=None, encoding='ISO-8859-1').astype(np.float32)
    DC_IV_dataframe.rename(columns = {0: 'voltage', 1: 'current', 2: 'resistance'}, inplace=True)
    DC_IV_dataframe['voltage'] = pd.Series([round(i, 2) for i in DC_IV_dataframe['voltage']])
    return DC_IV_dataframe

class Process_DC_IV():
     
    def __init__(self, sample_path: str) -> None:
        self.__sample_path = sample_path

    # выозвращает словарь со значениями токов во включенном и выключенном состоянии на основе списка измерений
    def on_off_current(self, dict_of_measurs: dict, check_voltage: float)-> dict:
        I_on = []
        I_off = []
        I_on_off = []
        for contact in list(dict_of_measurs.keys()):
            for measur in list(dict_of_measurs[contact].keys()):
                if dict_of_measurs[contact][measur] == 'DC_IV':
                    DC_IV_data = get_DC_IV_data(os.path.join(self.__sample_path, contact, measur + '.data'))
                else:
                    continue
                if check_voltage not in list(DC_IV_data['voltage']):
                    print(f'value V = {check_voltage} is not exist in file \'{measur}.data\' from \'{contact}\' contact')
                    continue
                else:
                    try:
                        a, b = DC_IV_data.loc[DC_IV_data['voltage'] == check_voltage]['current']
                        I_on.append(a)
                        I_off.append(b)
                        I_on_off.append(a/b)
                    except:
                        print(f'Unexpected error in file \'{measur}.data\' from \'{contact}\' folder')
                        continue
        return {'I_on': np.array(I_on), 'I_off':np.array(I_off), 'I_on_off': np.array(I_on_off)}

    # расчитывает напряжения включения и выключения у ВАХ типа ReRAM на основе списка измерений
    def ReRAM_on_off_voltage(self, dict_of_measurs: dict) -> np.array:
        on_off_voltage = []
        for contact in list(dict_of_measurs.keys()):
            for measur in list(dict_of_measurs[contact].keys()):
                if dict_of_measurs[contact][measur] == 'DC_IV':
                    DC_IV_data = get_DC_IV_data(os.path.join(self.__sample_path, contact, measur+ '.data'))
                else:
                    continue
                V, I = DC_IV_data['voltage'], DC_IV_data['current']
                delta_V = V[1] - V[0]
                deriv = np.array([(I[i+1] - I[i])/(V[i+1] - V[i]) if (V[i+1] - V[i]) != 0 else (I[i+1] - I[i])/delta_V for i in range(len(I) - 1) ])
                df_2 = pd.DataFrame([V[:-1], np.abs(deriv)]).transpose()
                df_2.rename(columns = {'voltage': 'voltage', 'Unnamed 0': 'derivative'}, inplace=True)
                df_plus = df_2.loc[df_2['voltage'] > 0].reset_index(drop = True)
                df_minus = df_2.loc[df_2['voltage'] < 0].reset_index(drop = True)
                on_off_voltage.append([df_minus.loc[np.argmax(df_minus['derivative'])]['voltage'] , np.abs(df_plus.loc[np.argmax(df_plus['derivative'])]['voltage'])])
        return np.array(on_off_voltage).transpose()
    
    # возвращает словать со всеми ВАХами на основе списка измерений
    def data_from_custom_dict(self, dict_of_measurs: dict) -> dict:
        data_dict = {}
        for contact in list(dict_of_measurs.keys()):
            single_contact_data = {}
            for measur in list(dict_of_measurs[contact].keys()):
                    if dict_of_measurs[contact][measur] == 'DC_IV':
                        single_contact_data[measur] = get_DC_IV_data(os.path.join(self.__sample_path, contact, measur + '.data'))
                    else:
                        continue
            data_dict[contact] = single_contact_data
        return data_dict

class Draw_DC_IV():

    def __init__(self, sample_path: str) -> None:
        self.__sample_path = sample_path
    
    def __to_colors(self, lenght: int, start_color: str, end_color: str):
        custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", [start_color, end_color])
        colors = custom_cmap(np.linspace(0, 1, lenght))
        return colors
        
    # рисует одну ВАХ
    def single_plot(self, contact: str, measur: str, save_path: str) -> None:
        DC_IV_data = get_DC_IV_data(os.path.join(self.__sample_path, str(contact), str(measur) + '.data'))
        V, I = DC_IV_data['voltage'], np.abs(DC_IV_data['current'])
        fig, ax = plt.subplots(figsize = [10,5])
        ax.set_yscale('log')
        ax.grid(which='major', linewidth = 0.6)
        ax.grid(which='minor', linewidth = 0.2)
        ax.set_xlim(xmin= V.min()*1.2, xmax=V.max()*1.2)
        ax.set_ylim(ymin= I.min()*0.2, ymax=I.max()*5)
        colors = colormaps['plasma'](np.linspace(0, 1, len(V)))
        segments = [[[V[i], I[i]], [V[i+1], I[i+1]]] for i in range(len(V)-1)]
        line_coll = LineCollection(segments)
        line_coll.set_color(colors)
        lines = ax.add_collection(line_coll)
        cbar = fig.colorbar(lines)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['start','end'], size = 15)
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 200)
        plt.close()

    # создает папку по запрошенному пути
    def _create_dir(self, path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            os.mkdir(path)
        else:
            os.mkdir(path)

    # рисует отдельные графиики по кастомному словарю и выбранному пути
    def from_dict(self, dict_of_measurs: dict, save_path: str) -> None:
        save_folder = os.path.join(os.path.dirname(self.__sample_path), str(save_path))
        self._create_dir(save_folder)
        for contact in list(dict_of_measurs.keys()):
            if len(list(dict_of_measurs.keys())) > 1:
                contact_save_path = os.path.join(save_folder, contact)
                self._create_dir(contact_save_path)
            else:
                contact_save_path = save_folder
            for measur in list(dict_of_measurs[contact].keys()):
                if dict_of_measurs[contact][measur] == 'DC_IV':
                    measure_save_path = os.path.join(contact_save_path, measur + '.png')
                    self.single_plot(contact, measur, measure_save_path)
                else:
                    continue

    # задает градиент для последовательности графиков
    def colorize_multiple(self, line_coll: LineCollection, start_color: str = '#ff0000', end_color: str = '#1e00ff'):
        line_coll.set_color(self.__to_colors(len(line_coll._paths), start_color, end_color))

    # рисует множество данных на одном графике
    def multiple(self, dict_of_measurs: dict, axes: axes, **kwargs) -> None:
        data_colletcion = []
        for contact in list(dict_of_measurs.keys()):
            for measur in list(dict_of_measurs[contact].keys()):
                if dict_of_measurs[contact][measur] == 'DC_IV':
                    DC_IV_data = get_DC_IV_data(os.path.join(self.__sample_path, contact, measur + '.data'))
                    data_colletcion.append(np.array([DC_IV_data['voltage'], np.abs(DC_IV_data['current'])]).transpose())
                else:
                    continue
        line_collection = LineCollection(data_colletcion, **kwargs)
        lines = axes.add_collection(line_collection)
        axes.autoscale_view()
        return lines
    
    # рисует всю инфу о ReRAM переключениях из отдельного словаря
    def full_ReRAM_info(self, dict_of_measurs: dict, I_on_off: dict, V_on_off: np.array, current_scale: int = 1000, bins: int = 10, start_color: str = '#ff0000', end_color: str = '#1e00ff', **kwargs):
        fig = plt.figure(figsize=(12, 7), constrained_layout=True)
        gs = GridSpec(ncols=10, nrows=4, figure=fig)

        ax_1 = plt.subplot(gs[:4, :6])

        ax_2_1 = plt.subplot(gs[0,6:9])
        ax_2_2 = plt.subplot(gs[1,6:9])
        ax_2_3 = plt.subplot(gs[2,6:9])
        ax_2_4 = plt.subplot(gs[3,6:9])

        ax_3_1 = plt.subplot(gs[0,9])
        ax_3_2 = plt.subplot(gs[1,9])
        ax_3_3 = plt.subplot(gs[2,9])
        ax_3_4 = plt.subplot(gs[3,9])

        coll = self.multiple(dict_of_measurs, ax_1, **kwargs)
        self.colorize_multiple(coll, start_color, end_color)
        ax_1.autoscale_view()
        ax_1.set_yscale('log')
        y_major = ticker.LogLocator(numticks = 10)
        y_minor = ticker.LogLocator(subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax_1.yaxis.set_major_locator(y_major)
        ax_1.yaxis.set_minor_locator(y_minor)

        ax_2_1.plot(range(len(I_on_off['I_on'])), I_on_off['I_on'] * current_scale)
        ax_2_2.plot(range(len(I_on_off['I_off'])), I_on_off['I_off'] * current_scale)
        ax_2_3.plot(range(len(V_on_off[0])), V_on_off[0])
        ax_2_4.plot(range(len(V_on_off[1])), V_on_off[1])

        ax_3_1.hist(I_on_off['I_on'] * current_scale, bins=bins, rwidth=0.8, edgecolor = 'k', orientation = 'horizontal')
        ax_3_1.axis('off')
        ax_3_1.set(ylim = ax_2_1.get_ylim())
        ax_3_2.hist(I_on_off['I_off'] * current_scale, bins=bins, rwidth=0.8, edgecolor = 'k', orientation = 'horizontal')
        ax_3_2.axis('off')
        ax_3_2.set(ylim = ax_2_2.get_ylim())
        ax_3_3.hist(V_on_off[0], bins=bins, rwidth=0.8, edgecolor = 'k', orientation = 'horizontal')
        ax_3_3.axis('off')
        ax_3_3.set(ylim = ax_2_3.get_ylim())
        ax_3_4.hist(V_on_off[1],bins=bins, rwidth=0.8,  edgecolor = 'k', orientation = 'horizontal')
        ax_3_4.axis('off')
        ax_3_4.set(ylim = ax_2_4.get_ylim())

        return (fig, ax_1, [ax_2_1, ax_2_2, ax_2_3, ax_2_4], [ax_3_1, ax_3_2, ax_3_3, ax_3_4])