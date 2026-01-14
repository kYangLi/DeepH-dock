from pathlib import Path
import re
from tqdm import tqdm

from joblib import Parallel, delayed

from deepx_dock.analyze.error.with_infer_res import ErrorElementsDistributionAnalyzer
from deepx_dock.analyze.error.with_infer_res import ErrorStructureDistributionAnalyzer
from deepx_dock.misc import load_json_file
from deepx_dock.CONSTANT import DEEPX_INFO_FILENAME

MASK_THRESHOLD = 1E-10


class ErrorElementsDistAnalyzerWithLog(ErrorElementsDistributionAnalyzer):
    def __init__(self,
        log_file_path, dft_dir, cache_res=False, parallel_num=1, 
    ):
        self.log_file_path = Path(log_file_path)
        self.dft_dir = dft_dir
        self.root_dir = self.log_file_path.parent
        self.cache_res = cache_res
        self.parallel_num = parallel_num
        #
        self.save_figure_path = self.root_dir / f'{self.TAG}.png'
        self.cached_result_path =  self.root_dir / f'{self.TAG}.json'
        # Results
        self.elem_error = None
    
    def analyze_all(self):
        if self.cached_result_path.is_file():
            self._load_cached_result()
            return
        results = self._get_all_results()
        self._postprocess_results(results)
        if self.cache_res:
            self._cache_the_result()
    
    def _get_all_results(self):
        result_dict = self._collect_res_from_logfile()
        results = Parallel(n_jobs=self.parallel_num)(
            delayed(self._get_one_result)(sid, error, self.dft_dir)
            for sid, error in tqdm(result_dict.items(), desc="Error Analysis")
        )
        return results
    
    def _collect_res_from_logfile(self):
        step_structure = {}
        result_dict = {}
        with open(self.log_file_path, 'r') as frp:
            for line in frp:
                if ('test-step' in line) and ('Structure ID(s):' in line):
                    step_match = re.search(r'Step (\d+)', line)
                    if step_match:
                        step = step_match.group(1)
                        structure_id = [
                            v for v in eval(line.split('Structure ID(s):')[-1])
                            if v != ''
                        ]
                        step_structure[step] = structure_id
                elif ('test-step' in line) and 'Curr-Loss' in line:
                    step_match = re.search(r'Step (\d+)', line)
                    if step_match:
                        step = step_match.group(1)
                        if step not in step_structure:
                            raise ValueError(f"The structure ID(s) of step {step} not found in log file.")
                        sids = step_structure[step]
                        loss_match = re.search(r'Curr-Loss (\S+)', line)
                        if loss_match:
                            error = float(loss_match.group(1).replace("|",""))
                            for sid in sids:
                                result_dict[sid] = error
        return result_dict
    
    @staticmethod
    def _get_one_result(sid: str, error: float, dft_dir: str | Path):
        dft_dir = Path(dft_dir)
        info_path = dft_dir / sid / DEEPX_INFO_FILENAME
        info = load_json_file(info_path)
        atoms_elem = list(info["elements_orbital_map"].keys())
        return {"elements": atoms_elem, "error": error}


class ErrorStructureDistributionAnalyzerWithLog(ErrorStructureDistributionAnalyzer):
    def __init__(self, log_file_path, cache_res=False):
        self.log_file_path = Path(log_file_path)
        self.cache_res = cache_res
        self.root_dir = self.log_file_path.parent
        #
        self.save_figure_path = self.root_dir / f'{self.TAG}.png'
        self.cached_result_path = self.root_dir / f'{self.TAG}.json'
        # Results
        self.errors = None
    
    def analyze_all(self):
        if self.cached_result_path.is_file():
            self._load_cached_result()
            return
        results = self._get_all_results()
        self._postprocess_results(results)
        if self.cache_res:
            self._cache_the_result()
    
    def _get_all_results(self):
        step_structure = {}
        results = []
        with open(self.log_file_path, 'r') as frp:
            for line in frp:
                if ('test-step' in line) and ('Structure ID(s):' in line):
                    step_match = re.search(r'Step (\d+)', line)
                    if step_match:
                        step = step_match.group(1)
                        structure_id = [
                            v for v in eval(line.split('Structure ID(s):')[-1])
                            if v != ''
                        ]
                        step_structure[step] = structure_id
                elif ('test-step' in line) and ('Curr-Loss' in line):
                    step_match = re.search(r'Step (\d+)', line)
                    if step_match:
                        step = step_match.group(1)
                        sids = step_structure.get(step, ['Unknown',])
                        loss_match = re.search(r'Curr-Loss (\S+)', line)
                        if loss_match:
                            error = float(loss_match.group(1).replace("|",""))
                            for sid in sids:
                                results.append((sid, error))
        return results

