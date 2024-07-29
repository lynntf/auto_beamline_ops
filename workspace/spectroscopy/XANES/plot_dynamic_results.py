import os
import glob
import pickle
import json
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
import pandas as pd
import scipy.interpolate
from torch import normal

from autobl.util import *
import autobl.tools.spectroscopy.xanes as xanestools
from plot_results import ResultAnalyzer
from XANES_grid_transfer_LTO import LTOGridTransferTester
from XANES_grid_transfer_Pt import PtGridTransferTester

import logging
logging.basicConfig(level=logging.CRITICAL)
logging.disable(logging.CRITICAL)


class DynamicExperimentResultAnalyzer:

    style_list = ResultAnalyzer.style_list

    def __init__(self, tester_class=LTOGridTransferTester, output_dir='factory'):
        self.output_dir = output_dir
        self.tester_class = tester_class

        matplotlib.rc('font', family='Times New Roman')
        matplotlib.rcParams['font.size'] = 12
        matplotlib.rcParams['pdf.fonttype'] = 42

    @staticmethod
    def get_metadata(folder):
        with open(os.path.join(folder, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        return metadata

    def plot_data(self, folder, transpose=False, xtick_interval=None, value_range=(0, 1.7), normalizer=None, plot_raw_data=False,
                  plot_func='imshow', line_plot_labels=None, output_filename='test_data.pdf'):
        metadata = self.get_metadata(folder)
        tester = self.tester_class(**metadata)
        tester.load_data(normalizer=normalizer)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        if plot_raw_data:
            x, y = to_numpy(tester.test_energies_raw.squeeze()), np.arange(len(tester.test_data_all_spectra))
            all_spectra = tester.test_data_all_spectra_raw.copy()
        else:
            x, y = to_numpy(tester.test_energies.squeeze()), np.arange(len(tester.test_data_all_spectra))
            all_spectra = tester.test_data_all_spectra.copy()

        if plot_func == 'imshow':
            if transpose:
                x, y = y, x
                all_spectra = all_spectra.T
            pts = np.stack(np.meshgrid(y, x, indexing='ij'), axis=-1).reshape(-1, 2)
            xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)))
            zz = scipy.interpolate.griddata(pts, all_spectra.reshape(-1), (yy, xx), 'linear')
            if value_range is None:
                value_range = (None, None)
            im = ax.imshow(zz, extent=[x.min(), x.max(), y.max(), y.min()], vmin=value_range[0], vmax=value_range[1],
                           cmap='jet')
            if xtick_interval is not None:
                ax.set_xticks(np.arange(x.min(), x.max(), xtick_interval))
            plt.colorbar(im, fraction=0.046, pad=0.04)
            if transpose:
                ax.set_ylabel('Energy (eV)')
                ax.set_xlabel('Spectrum index')
            else:
                ax.set_xlabel('Energy (eV)')
                ax.set_ylabel('Spectrum index')
        elif plot_func == 'plot':
            if not plot_raw_data:
                all_spectra = np.concatenate([tester.ref_data_all_spectra[0][None, :], tester.test_data_all_spectra, tester.ref_data_all_spectra[-1][None, :]])
            else:
                all_spectra = np.concatenate([tester.ref_data_all_spectra_raw[0][None, :], tester.test_data_all_spectra_raw, tester.ref_data_all_spectra_raw[-1][None, :]])
            fig, ax = plt.subplots(figsize=(8, 5))
            cmap_list = matplotlib.colormaps['jet'](np.linspace(0, 1, all_spectra.shape[0]))
            for i in range(all_spectra.shape[0]):
                lab = line_plot_labels[i] if line_plot_labels is not None else None
                ax.plot(x, all_spectra[i], label=lab, linewidth=0.5, color=cmap_list[i])
            if line_plot_labels is not None:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10)
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Spectrum index')
            plt.grid()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))

        # Beginning and end spectra
        for name, dset in (('spectra', tester.test_data_all_spectra), ('ref_spectra', tester.ref_data_all_spectra)):
            x, y = to_numpy(tester.test_energies.squeeze()), dset
            fig, ax = plt.subplots(2, 1, figsize=(4, 4))
            ax[0].plot(x, y[0])
            ax[0].set_ylabel('Normalized\nx-ray absorption')
            ax[0].grid()
            ax[1].plot(x, y[-1])
            ax[1].set_ylabel('Normalized\nx-ray absorption')
            ax[1].set_xlabel('Energy (eV)')
            ax[1].grid()
            plt.tight_layout()
            fig.savefig(os.path.join(self.output_dir, os.path.splitext(output_filename)[0] + '_{}.pdf'.format(name)))

    def compare_rms_across_time(self, result_folders, labels, output_filename='comparison_rms_across_time.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        for i, folder in enumerate(result_folders):
            rms_list = np.loadtxt(os.path.join(folder, 'rms_all_test_spectra.txt'))
            ax.plot(rms_list, linestyle=self.style_list[i], label=labels[i])
        ax.set_xlabel('Spectrum index')
        ax.set_ylabel('RMS')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))

    def calculate_phase_transition_percentages(self, folder):
        metadata = self.get_metadata(folder)
        tester = self.tester_class(**metadata)
        tester.load_data()
        indices = []
        percentages_estimated = []
        percentages_true = []
        fitting_residue_estimated = []
        flist = glob.glob(os.path.join(self.output_dir, 'estimated_data_ind_*.csv'))
        flist = np.array(flist)[np.argsort([int(re.findall('\d+', x)[-1]) for x in flist])]
        for f in flist:
            ind = int(re.findall('\d+', f)[-1])
            indices.append(ind)
            table_spectrum = pd.read_csv(f, index_col=None)
            estimated_spectrum = table_spectrum['estimated_data'].to_numpy()
            true_spectrum = table_spectrum['true_data'].to_numpy()
            p_estimated, r = tester.get_phase_transition_percentage(estimated_spectrum, return_fitting_residue=True,
                                                                    add_constant_term=True)
            percentages_estimated.append(p_estimated)
            fitting_residue_estimated.append(r)
            p_true, r = tester.get_phase_transition_percentage(true_spectrum, return_fitting_residue=True,
                                                               add_constant_term=True)
            percentages_true.append(p_true)
        table = pd.DataFrame(data={'indices': indices, 'percentages_estimated': percentages_estimated,
                                   'percentages_true': percentages_true})
        return table

    def compare_calculated_percentages(self, result_folders, labels, x_data=None, x_label='Spectrum index', 
                                       read_precalculated_percentage_data=True, normalizer=None,
                                       output_filename='comparison_calculated_percentages.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.8))
        table = {}
        for i, folder in enumerate(result_folders):
            if read_precalculated_percentage_data:
                table = pd.read_csv(os.path.join(folder, 'phase_transition_percentages.csv'))
            else:
                metadata = self.get_metadata(folder)
                tester = self.tester_class(**metadata)
                tester.load_data(normalizer=normalizer)
                table = tester.calculate_phase_transition_percentages(normalizer=normalizer)
            if x_data is None:
                x_data = table['indices']
            ax.plot(x_data, np.clip(table['percentages_estimated'] * 100, 0, 100), linestyle=self.style_list[i], label=labels[i])
        ax.plot(x_data, np.clip(table['percentages_true'] * 100, 0, 100), linestyle='--', color='gray', label='Ground truth')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Phase transition percentage (%)')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))

    def plot_spectrum(self, result_folders, labels, spectrum_index, normalizer=None, output_filename="spectrum.pdf"):
        if normalizer is not None:
            metadata = self.get_metadata(result_folders[0])
            tester = self.tester_class(**metadata)
            tester.load_data(normalizer=normalizer)
            normalizer.fit(tester.test_energies_raw, tester.test_data_all_spectra_raw[spectrum_index])
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        for i, (result_folder, label) in enumerate(zip(result_folders, labels)):
            table = pd.read_csv(os.path.join(result_folder, 'estimated_data_ind_{}.csv'.format(spectrum_index)))
            energies = table['energy']
            estimated_data = table['estimated_data']
            true_data = table['true_data']
            if normalizer is not None:
                estimated_data = normalizer.apply(energies, estimated_data)
                true_data = normalizer.apply(energies, true_data)
            if i == 0:
                ax.plot(energies, true_data, color='gray', linestyle='--', label='Ground truth')
            ax.plot(energies, estimated_data, linestyle=self.style_list[i], label=label)
        ax.grid()
        ax.legend()
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Normalized\nx-ray absorption')
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))


if __name__ == '__main__':
    folders = [
        'outputs/grid_transfer_LTO/grid_redoForEach/50C',
        'outputs/grid_transfer_LTO/grid_initOfSelf/50C',
        'outputs/grid_transfer_LTO/grid_selectedRef/50C',
    ]
    labels = [
        'Run for each',
        'Initial spectrum',
        'Reference spectra'
    ]
    
    normalizer = xanestools.XANESNormalizer(fit_ranges=((4900, 4950), (5050, 5200)), edge_loc=4983)
    
    analyzer = DynamicExperimentResultAnalyzer(tester_class=LTOGridTransferTester)
    analyzer.plot_data(folders[0], transpose=True, value_range=(None, None), normalizer=normalizer, plot_func='imshow', line_plot_labels=list(range(200)), output_filename='LTO_50C_test_data.pdf')
    analyzer.plot_data(folders[0], transpose=True, value_range=(None, None), plot_raw_data=True, plot_func='imshow', line_plot_labels=list(range(200)), output_filename='LTO_50C_test_data_raw.pdf')
    analyzer.plot_data('outputs/grid_transfer_LTO/grid_initOfSelf/70C', transpose=True, value_range=(None, None), normalizer=normalizer, plot_func='imshow', line_plot_labels=list(range(200)), xtick_interval=5, output_filename='LTO_70C_test_data.pdf')
    analyzer.plot_data('outputs/grid_transfer_LTO/grid_initOfSelf/70C', transpose=True, value_range=(None, None), plot_raw_data=True, plot_func='imshow', line_plot_labels=list(range(200)), xtick_interval=5, output_filename='LTO_70C_test_data_raw.pdf')
    analyzer.compare_rms_across_time(folders, labels, output_filename='LTO_50C_comparison_rms_across_time.pdf')
    analyzer.compare_calculated_percentages(folders, labels, normalizer=normalizer, 
                                            read_precalculated_percentage_data=False, output_filename='LTO_50C_comparison_calculated_percentages.pdf')
    analyzer.plot_spectrum(folders, labels, 80, output_filename='LTO_50C_estimated_spectra_80.pdf', normalizer=normalizer)
    analyzer.plot_spectrum(folders, labels, 90, output_filename='LTO_50C_estimated_spectra_90.pdf', normalizer=normalizer)

    # -----------------------------------

    folders = [
        'outputs/grid_transfer_Pt/grid_redoForEach/Pt',
        'outputs/grid_transfer_Pt/grid_initOfSelf/Pt',
        'outputs/grid_transfer_Pt/grid_selectedRef/Pt',
    ]
    labels = [
        'Run for each',
        'Initial spectrum',
        'Reference spectra'
    ]

    normalizer = xanestools.XANESNormalizer(fit_ranges=((11400, 11500), (11650, 11850)), edge_loc=11566.0)
    
    analyzer = DynamicExperimentResultAnalyzer(tester_class=PtGridTransferTester)
    label_val_list = [int(re.findall('\d+', f)[-1]) for f in glob.glob('data/raw/Pt-XANES/Pt_xmu/*.xmu')]
    label_val_list.sort()
    label_list = [str(x) + '$^\circ\!$C' for x in label_val_list]
    analyzer.plot_data(folders[0], transpose=True, plot_func='plot', line_plot_labels=label_list, normalizer=normalizer, output_filename='Pt_grid_transfer_test_data.pdf')
    analyzer.plot_data(folders[0], transpose=True, plot_func='plot', line_plot_labels=label_list, normalizer=None, output_filename='Pt_grid_transfer_test_data_raw.pdf')
    analyzer.compare_rms_across_time(folders, labels, output_filename='Pt_grid_transfer_comparison_rms_across_time.pdf')
    analyzer.compare_calculated_percentages(folders, labels, x_data=label_val_list[1:-1], x_label='Temperature ($\!^\circ\!$C)', 
                                            read_precalculated_percentage_data=False, normalizer=normalizer, output_filename='Pt_grid_transfer_comparison_calculated_percentages.pdf')
    # analyzer.plot_spectrum(folders, labels, 0, output_filename='Pt_grid_transfer_estimated_spectra_0.pdf')
    # analyzer.plot_spectrum(folders, labels, 21, output_filename='Pt_grid_transfer_estimated_spectra_21.pdf')
