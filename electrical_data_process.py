import matplotlib
from matplotlib import pyplot as plt
import matplotlib.axes
from matplotlib.collections import LineCollection
import matplotlib.collections
import pandas as pd 
import numpy as np 
import os
import shutil

class Measurments():
    # инициализация, проверка существования папки, проверка пустоты папки
    def __init__(self, main_folder: str) -> None:
        if not os.path.exists(main_folder):
            raise FileNotFoundError(f'directory \'{main_folder}\' is not exist')
        elif  (len(os.listdir(main_folder)) == 0):
            raise ValueError(f'directory \'{main_folder}\' is empty')
        else:
            self.main_folder = main_folder
            self.all_files_list = os.listdir(main_folder)
            self.main_file_path = os.path.abspath(main_folder)
            self._create_dict_of_measurments()
            
    # поиск всех вложенных директорий в основной папке
    def _find_contacts_folder(self) -> None:
        self.contacts_list = set()
        for i in self.all_files_list:
            if os.path.isdir('\\'.join([self.main_folder, i])):
                self.contacts_list.add(f'{i}')

    # проверка, что файл типа data
    def _is_data_file(self, path: str) -> bool:
        if path.split('.')[-1] == 'data':
            return True
        else:
            return False
    # создание пути из названия папок
    def _create_path(self, list: list[str]) -> str:
        return '\\'.join([self.main_file_path] + list)

    # считывание данных об измерениях из файлов во вложенных директориях
    def _create_dict_of_measurments(self, return_dict: bool = False) -> None | dict:
        self._find_contacts_folder()
        if len(self.contacts_list) == 0:
            raise ValueError(f'directory \'{self.main_folder}\' does not contain subdirectories')
        self.dict_of_measurments = {}
        for i in self.contacts_list:
            subdir_catalogs_list = [file for file in os.listdir(self._create_path([i])) if self._is_data_file(self._create_path([i, file]))]
            if len(subdir_catalogs_list) == 0:
                continue
            contact_measurs_type = {}
            for j in subdir_catalogs_list:
                file_path = self._create_path([i, j])
                with open(file_path) as file:
                    measur_type = '_'.join(file.readlines()[1].split()[1:3])
                contact_measurs_type[j.replace('.data', '')] = measur_type
            self.dict_of_measurments[i] = contact_measurs_type
        if return_dict == True:
            return self.dict_of_measurments
        
    # возврат словаря с измерениями
    def get_dict_of_measurments(self)-> dict:
        if hasattr(self, 'dict_of_measurments'):
            return self.dict_of_measurments
        else:
            self._create_dict_of_measurments()
            return self.dict_of_measurments
        
    # абсолютный путь к указанной папке
    def get_abspath(self) -> str:
        return self.main_file_path
    
    # получение словаря только с одного контакта
    def get_contact_dict(self, contact_name: str | int) -> dict:
        if not isinstance(contact_name, str | int):
            raise ValueError(f'contact_name must be str or int type')
        elif not hasattr(self, 'dict_of_measurments'):
            self._create_dict_of_measurments()
        elif str(contact_name) not in list(self.dict_of_measurments.keys()):
            raise ValueError(f'{contact_name} not exist in {self.main_folder}')
        return {str(contact_name) :self.dict_of_measurments[str(contact_name)]}
    
    # удаление папки или папок
    def delete_contacts(self, contact_name: str | list, return_dict: bool = False) -> dict | None:
        if not hasattr(self, 'dict_of_measurments'):
            self._create_dict_of_measurments()
        contact_from_dict = list(self.dict_of_measurments.keys())
        existed_contact = []
        # проверка типов и существования
        if not isinstance(contact_name, list):
            contact_name = [contact_name]
        for i in contact_name:
            if not isinstance(i, str):
                try:
                    i = str(i)
                except:
                    raise TypeError(f'{i} is not {str} or can\'t be converted to {str}')
            if i in contact_from_dict:
                existed_contact.append(i)
        for i in existed_contact:
            del self.dict_of_measurments[i]
        if return_dict == True:
            return self.dict_of_measurments
    
    # удаление измерений из контакта 
    def delete_measurments(self, del_dict: dict) -> None:
        if not hasattr(self, 'dict_of_measurments'):
            self._create_dict_of_measurments()
        del_dict_keys = list(del_dict.keys())
        all_dict_keys = list(self.dict_of_measurments.keys())
        for contact in del_dict_keys:
            if str(contact) in all_dict_keys:
                if not isinstance(del_dict[contact], list):
                    raise ValueError(f'dict values must be {list} type')
                for measur in del_dict[contact]:
                    if str(measur) in list(self.dict_of_measurments[str(contact)]):
                        self.dict_of_measurments[str(contact)].pop(str(measur), None)
                    else:
                        continue
            else:
                continue

class Process_DC_IV():
     
    def __init__(self, sample_path: str) -> None:
        self.sample_path = sample_path

    # возвращает датафрейм с ВАХ конкретного измерения
    def get_single_data(self, contact: str, measure: str) -> pd.DataFrame:
        measure_dir = self.sample_path + '\\' + str(contact) + '\\' + str(measure) + '.data'
        dataframe = pd.read_csv(measure_dir, delimiter='   ', skiprows=16, engine='python', header=None, encoding='ISO-8859-1').astype(np.float32)
        dataframe.rename(columns = {0: 'voltage', 1: 'current', 2: 'resistance'}, inplace=True)
        dataframe['voltage'] = pd.Series([round(i, 2) for i in dataframe['voltage']])
        return dataframe

    # выозвращает словарь со значениями токов во включенном и выключенном состоянии на основе списка измерений
    def get_on_off_current(self, dict_of_measurs: dict, check_voltage: float)-> dict:
        I_on = []
        I_off = []
        I_on_off = []
        for folder in list(dict_of_measurs.keys()):
            for measur in list(dict_of_measurs[folder].keys()):
                if dict_of_measurs[folder][measur] == 'DC_IV':
                    DC_IV_data = self.get_single_data(folder, measur)
                else:
                    continue
                if check_voltage not in list(DC_IV_data['voltage']):
                    print(f'value V = {check_voltage} is not exist in file \'{measur}.data\' from \'{folder}\' folder')
                    continue
                else:
                    try:
                        a, b = DC_IV_data.loc[DC_IV_data['voltage'] == check_voltage]['current']
                        if a > b:
                            I_on.append(a)
                            I_off.append(b)
                            I_on_off.append(a/b)
                        else:
                            I_on.append(b)
                            I_off.append(a)
                            I_on_off.append(b/a)
                    except:
                        print(f'Unexpected error in file \'{measur}.data\' from \'{folder}\' folder')
                        continue
        return {'I_on': I_on, 'I_off': I_off, 'I_on_off': I_on_off}

    # расчитывает напряжения включения и выключения у ВАХ типа ReRAM на основе списка измерений
    def ReRAM_on_off_voltage(self, dict_of_measurs: dict) -> np.array:
        on_off_voltage = []
        for folder in list(dict_of_measurs.keys()):
            for measur in list(dict_of_measurs[folder].keys()):
                if dict_of_measurs[folder][measur] == 'DC_IV':
                    DC_IV_data = self.get_single_data(folder, measur)
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
    def get_data_from_custom_dict(self, dict_of_measurs: dict) -> dict:
        data_dict = {}
        for folder in list(dict_of_measurs.keys()):
            single_contact_data = {}
            for measur in list(dict_of_measurs[folder].keys()):
                    if dict_of_measurs[folder][measur] == 'DC_IV':
                        single_contact_data[measur] = self.get_single_data(folder, measur)
                    else:
                        continue
            data_dict[folder] = single_contact_data
        return data_dict

class Draw_DC_IV(Process_DC_IV):

    def __init__(self, process: Process_DC_IV, measures: Measurments) -> None:
        if isinstance(process, Process_DC_IV) and isinstance(measures, Measurments):
            self.sample_path = process.sample_path
            self.dict_of_measurments = measures.dict_of_measurments
            self.sample_path = measures.get_abspath()
            self.main_folder = measures.main_folder
        else:
            ValueError(f'input values must be {Process_DC_IV} and {Measurments} type')

    # последовательно раскрашивает линию на графике от начала в конец
    def _colored_line(self, voltage, current, c, ax: matplotlib.axes, **lc_kwargs):
        default_kwargs = {"capstyle": "butt"}
        default_kwargs.update(lc_kwargs)
        x = np.asarray(voltage)
        y = np.asarray(current)
        x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
        y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
        coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
        coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
        coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
        segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
        lc = LineCollection(segments, **default_kwargs)
        lc.set_array(c) 
        return ax.add_collection(lc)
        
    # рисует одну ВАХ
    def single_plot(self, contact: str, measure: str, save_path: str) -> None:
        DC_IV_data = self.get_single_data(contact=contact, measure=measure)
        V, I = DC_IV_data['voltage'], np.abs(DC_IV_data['current'])
        fig, ax = plt.subplots(figsize = [10,5])
        ax.set_title(f'Contact {contact}, measurement № {measure}')
        ax.set_yscale('log')
        ax.grid(which='major', linewidth = 0.6)
        ax.grid(which='minor', linewidth = 0.2)
        ax.set_xlim(xmin= V.min()*1.2, xmax=V.max()*1.2)
        ax.set_ylim(ymin= I.min()*0.2, ymax=I.max()*5)
        color = np.linspace(0, 1, len(V))
        lines = self._colored_line(V, I, color, ax, cmap = 'plasma')
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
        save_folder = os.path.dirname(self.sample_path) + '\\' + str(save_path)
        self._create_dir(save_folder)
        for contact in list(dict_of_measurs.keys()):
            contact_save_path = save_folder + '\\' + contact
            self._create_dir(contact_save_path)
            for measur in list(dict_of_measurs[contact].keys()):
                if dict_of_measurs[contact][measur] == 'DC_IV':
                    measure_save_path = contact_save_path + '\\' + measur + '.png'
                    self.single_plot(contact, measur, measure_save_path)
                else:
                    continue

    # рисует все графики 
    def all(self):
        self.from_dict(self.dict_of_measurments, self.main_folder + '_graphs')

    # переводит hex значения цветов в RGB
    def _hex_to_RGB(self, hex_str):
        return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

    # задает градиент для последовательности графиков
    def colored_lines(self, line_coll: matplotlib.collections.LineCollection, start_color: str = '#ff0000', end_color: str = '#1e00ff'):
        n = len(line_coll._paths)
        start_color_rgb = np.array(self._hex_to_RGB(start_color))/255
        end_color_rgb = np.array(self._hex_to_RGB(end_color))/255
        mix_pcts = [x/(n-1) for x in range(n)]
        rgb_colors = [((1-mix)*start_color_rgb + (mix*end_color_rgb)) for mix in mix_pcts]
        colors = ['#' + ''.join([format(int(round(val*255)), '02x') for val in item]) for item in rgb_colors]
        line_coll.set_color(colors)

    # рисует множество данных на одном графике
    def multiple(self, dict_of_measurs: dict, axes: matplotlib.axes, **kwargs) -> None:
        data_colletcion = []
        for folder in list(dict_of_measurs.keys()):
            sorted_measurs_dict = dict(sorted(dict_of_measurs[folder].items(), key=lambda item: int(item[0])))
            for measur in list(sorted_measurs_dict.keys()):
                if dict_of_measurs[folder][measur] == 'DC_IV':
                    DC_IV_data = self.get_single_data(folder, measur)
                    data_colletcion.append(np.array([DC_IV_data['voltage'], np.abs(DC_IV_data['current'])]).transpose())
                else:
                    continue
        line_collection = LineCollection(data_colletcion, **kwargs)
        return axes.add_collection(line_collection, autolim=True)