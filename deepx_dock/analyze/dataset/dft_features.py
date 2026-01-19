import logging
from typing import Dict, List, Any
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, asdict

import warnings
from functools import partial
from joblib import Parallel, delayed

import numpy as np

from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL
from deepx_dock.CONSTANT import PERIODIC_TABLE_SYMBOL_TO_INDEX
from deepx_dock.CONSTANT import DEEPX_INFO_FILENAME
from deepx_dock.misc import dump_json_file, load_json_file
from deepx_dock.misc import get_data_dir_lister, list_A_contained_B

DATASET_FEATURES_FILENAME = "features_preview.json"


@dataclass
class DFTDatasetFeatures:
    # Features condition
    _ready_to_be_used: bool = False
    # System related features
    root_path: Path | None = None
    all_dft_data_num: int | None = None
    all_dft_dirname: List[str] | None = None
    # H like related task features
    elements_orbital_map: Dict[str, List[int]] | None = None
    common_orbital_types: List[int] | None = None
    common_orbital_num: int | None = None
    spinful: bool | None = None
    # Basis fitting related task features
    elements_fitting_map: Dict[str, List[int]] | None = None
    common_fitting_types: List[int] | None = None
    common_fitting_num: int | None = None
    # Real grid related task features
    max_Vr_size: int | None = None
    # Force field related task features
    elements_force_rcut_map: Dict[str, float] | None = None
    max_num_neighbors: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Special treatment for some keys
        del data['root_path']
        data['elements_orbital_map'] = {
            PERIODIC_TABLE_INDEX_TO_SYMBOL[k]: v
            for k, v in data['elements_orbital_map'].items()
        } if data['elements_orbital_map'] is not None else None
        data['elements_fitting_map'] = {
            PERIODIC_TABLE_INDEX_TO_SYMBOL[k]: v
            for k, v in data['elements_fitting_map'].items()
        } if data['elements_fitting_map'] is not None else None
        data['elements_force_rcut_map'] = {
            PERIODIC_TABLE_INDEX_TO_SYMBOL[k]: v
            for k, v in data['elements_force_rcut_map'].items()
        } if data['elements_force_rcut_map'] is not None else None
        # Return
        return data

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], root_path: str | Path
    ) -> 'DFTDatasetFeatures':
        # Special treatment for some keys
        data['root_path'] = Path(root_path)
        data['elements_orbital_map'] = {
            PERIODIC_TABLE_SYMBOL_TO_INDEX[k]: v
            for k, v in data['elements_orbital_map'].items()
        } if data.get('elements_orbital_map') is not None else None
        data['elements_fitting_map'] = {
            PERIODIC_TABLE_SYMBOL_TO_INDEX[k]: v
            for k, v in data['elements_fitting_map'].items()
        } if data.get('elements_fitting_map') is not None else None
        data['elements_force_rcut_map'] = {
            PERIODIC_TABLE_SYMBOL_TO_INDEX[k]: v
            for k, v in data['elements_force_rcut_map'].items()
        } if data.get('elements_force_rcut_map') is not None else None
        # Return class!
        return cls(
            root_path=data['root_path'],
            all_dft_data_num=data['all_dft_data_num'],
            all_dft_dirname=data['all_dft_dirname'],
            #
            elements_orbital_map=data.get('elements_orbital_map'),
            common_orbital_types=data.get('common_orbital_types'),
            common_orbital_num=data.get('common_orbital_num'),
            spinful=data.get('spinful'),
            #
            elements_fitting_map=data.get('elements_fitting_map'),
            common_fitting_types=data.get('common_fitting_types'),
            common_fitting_num=data.get('common_fitting_num'),
            #
            max_Vr_size=data.get('max_Vr_size'),
            #
            elements_force_rcut_map=data.get('elements_force_rcut_map'),
            max_num_neighbors=data.get('max_num_neighbors')
        )

    def dump(self, file_path: Path) -> None:
        data_dict = {k: v for k, v in self.to_dict().items() if v is not None}
        dump_json_file(file_path, data_dict)

    @classmethod
    def load(cls, file_path: Path) -> 'DFTDatasetFeatures':
        file_path = Path(file_path)
        root_path = file_path.parent
        data = load_json_file(file_path)
        return cls.from_dict(data, root_path)

    @property
    def ready_to_be_used(self):
        return self._ready_to_be_used

    def claim_ready_to_be_used(self):
        self._ready_to_be_used = True


class DFTDatasetFeaturesDetective:
    def __init__(self, 
        dft_path: str | Path, planned_common_orbital_types: List | None = None,
        data_dir_depth: int = 0, parallel_num: int = -1, 
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.root_path = Path(dft_path)
        self.planned_common_orbital_types = planned_common_orbital_types
        self.data_dir_depth = data_dir_depth
        self.parallel_num = parallel_num
        #
        self.feature_file = self.root_path / DATASET_FEATURES_FILENAME
        self._features = DFTDatasetFeatures()

    @property
    def features(self):
        if self._features.ready_to_be_used:
            return self._features
        # Read Basic Info
        if not self.feature_file.is_file():
            self.logger.info(
                f"[rawdata] Collecting features from raw DFT data: `{self.root_path}`."
            )
            self._collect_features_from_dft_raw_data()
            self._features.dump(self.feature_file)
        else:
            self.logger.info(
                f"[rawdata] Read in features from json: `{self.feature_file}`."
            )
            self._features = self._features.load(self.feature_file)
        # Post-process DFT data features
        self._post_process_dft_features()
        # Return!
        self._features.claim_ready_to_be_used()
        return self._features

    def _collect_features_from_dft_raw_data(self):
        self._find_all_dft_data_dir()
        self._get_info_from_whole_dft_raw_data()
        self._try_to_decided_common_orbital_types()
        self._try_to_decided_common_fitting_types()
    
    def _post_process_dft_features(self):
        self._try_to_consider_planned_common_orbital_types()
    
    def _try_to_consider_planned_common_orbital_types(self):
        # If the no planned common orbital types set, skip this function.
        planned_comm_orb_types = self.planned_common_orbital_types
        if (not planned_comm_orb_types) or planned_comm_orb_types is None:
            self.logger.debug("[rawdata] The common orbital types are read from DFT data.")
            return
        # Determine if the planned common orbital types are valid to use.
        dft_comm_orb_types = self._features.common_orbital_types
        if not list_A_contained_B(planned_comm_orb_types, dft_comm_orb_types):
            raise ValueError(
                f"The given planned common orbital types '{planned_comm_orb_types}' are insufficient to represent the DFT data '{dft_comm_orb_types}'. Please check settings!"
            )
        self.logger.debug(
            "[rawdata] The common orbital types are read as user planned."
        )
        self._features.common_orbital_types = planned_comm_orb_types
        self._features.common_orbital_num = \
            self._get_common_num(planned_comm_orb_types)

    def _find_all_dft_data_dir(self):
        print("[do] Locate all DFT data directories ...", flush=True)
        self.logger.info(f"[rawdata] Processing DFT data in `{self.root_path}`.")
        lister = get_data_dir_lister(self.root_path, self.data_dir_depth)
        all_dft_dir_list = [str(d) for d in tqdm(lister, desc="  +-[search]")]
        #
        structures_num = len(all_dft_dir_list)
        if structures_num <= 0:
            raise FileNotFoundError(f"[error] No valid data found in `{self.root_path}`")
        self.logger.info(f"[rawdata] Found `{structures_num}` structures in `{self.root_path}`.")
        #
        self._features.root_path = self.root_path
        self._features.all_dft_dirname = sorted(all_dft_dir_list)
        self._features.all_dft_data_num = structures_num

    def _get_info_from_whole_dft_raw_data(self):
        print("[do] Parsing DFT data features ...", flush=True)
        self.logger.info(
            f"[rawdata] Parsing elements of materials one by one with {self.parallel_num} processes..."
        )
        # Get the unsorted element-orbital map
        worker = partial(
            self._try_expand_one_dft_data_features,
            root_path=self.root_path,
        )
        warnings.filterwarnings("ignore", message=r"os\.fork\(\) was called.")
        results = Parallel(n_jobs=self.parallel_num)(
            delayed(worker)(d)
            for d in tqdm(self._features.all_dft_dirname, desc="  +-[parse]")
        )
        self._summarize_info(results)
    
    def _summarize_info(self, results):
        # Summary the results
        _one_result = results[0]
        _elements_orbital_map = _one_result["elem_orb_map"]
        _elements_fitting_map = _one_result["elem_fitting_map"]
        _elements_force_rcut_map = _one_result["elem_force_rcut_map"]
        _spinful = _one_result["spinful"]
        _max_Vr_size = _one_result["Vr_size"]
        _max_num_neighbors = _one_result["max_num_neighbors"]
        for res in tqdm(results, desc="  +-[summarize]"):
            #
            if _elements_orbital_map is not None:
                for elem, orb in res["elem_orb_map"].items():
                    if elem in _elements_orbital_map:
                        assert _elements_orbital_map[elem] == orb
                    else:
                        _elements_orbital_map[elem] = orb
            #
            if _elements_fitting_map is not None:
                for elem, orb in res["elem_fitting_map"].items():
                    if elem in _elements_fitting_map:
                        assert _elements_fitting_map[elem] == orb
                    else:
                        _elements_fitting_map[elem] = orb
            #
            if _elements_force_rcut_map is not None:
                for elem, rcut in res["elem_force_rcut_map"].items():
                    _elements_force_rcut_map[elem] = max(
                        _elements_force_rcut_map.get(elem, 0.0), rcut
                    )
            # 
            if _spinful is not None:
                assert _spinful == res["spinful"]
            #
            if _max_num_neighbors is not None:
                assert _max_num_neighbors == res["max_num_neighbors"]
            #
            if _max_Vr_size is not None:
                _max_Vr_size = max(_max_Vr_size, res["Vr_size"])
        # Save to the feature context
        self._features.elements_orbital_map = {
            k: _elements_orbital_map[k]
            for k in sorted(_elements_orbital_map.keys())
        } if _elements_orbital_map is not None else None
        self._features.elements_fitting_map = {
            k: _elements_fitting_map[k]
            for k in sorted(_elements_fitting_map.keys())
        } if _elements_fitting_map is not None else None
        self._features.elements_force_rcut_map = {
            k: _elements_force_rcut_map[k]
            for k in sorted(_elements_force_rcut_map.keys())
        } if _elements_force_rcut_map is not None else None
        self._features.spinful = _spinful
        self._features.Vr_size = _max_Vr_size
        self._features.max_num_neighbors = _max_num_neighbors
    
    def _try_to_decided_common_orbital_types(self):
        self._features.common_orbital_types = \
            self._get_common_types(self._features.elements_orbital_map)
        self._features.common_orbital_num = \
            self._get_common_num(self._features.common_orbital_types)

    def _try_to_decided_common_fitting_types(self):
        self._features.common_fitting_types = \
            self._get_common_types(self._features.elements_fitting_map)
        self._features.common_fitting_num = \
            self._get_common_num(self._features.common_fitting_types)

    @staticmethod
    def _try_expand_one_dft_data_features(dft_dir: str, root_path: Path):
        try:
            return DFTDatasetFeaturesDetective._expand_one_dft_data_features(
                dft_dir, root_path
            )
        except Exception as e:
            raise ValueError(f"[Error] dft_dir {dft_dir}: {e}")

    @staticmethod
    def _expand_one_dft_data_features(dft_dir: str, root_path: Path):
        # Read!
        _abs_dft_path = root_path / dft_dir
        info_json_path = _abs_dft_path / DEEPX_INFO_FILENAME
        info = load_json_file(info_json_path)
        # - Orb map
        curr_elem_orb_map = {
            PERIODIC_TABLE_SYMBOL_TO_INDEX[k]: v
            for k,v in info["elements_orbital_map"].items()
        } if info.get("elements_orbital_map") is not None else None
        # - Fitting map
        curr_elem_fitting_map = {
            PERIODIC_TABLE_SYMBOL_TO_INDEX[k]: v
            for k,v in info["elements_fitting_map"].items()
        } if info.get("elements_fitting_map") is not None else None
        # - Elements cutoff radius
        curr_elem_force_r_cutoff_map = {
            PERIODIC_TABLE_SYMBOL_TO_INDEX[k]: v
            for k,v in info["elements_force_rcut_map"].items()
        } if info.get("elements_force_rcut_map") is not None else None
        # - Spinful
        spinful = info.get("spinful")
        # - Vr size
        Vr_size = info.get("Vr_size")
        # - Max numbers of neighbors
        max_num_neighbors = info.get("max_num_neighbors")
        return {
            "elem_orb_map": curr_elem_orb_map,
            "elem_fitting_map": curr_elem_fitting_map,
            "elem_force_rcut_map": curr_elem_force_r_cutoff_map,
            "spinful": spinful,
            "Vr_size": Vr_size,
            "max_num_neighbors": max_num_neighbors,
        }

    @staticmethod
    def _get_common_types(elements_orbital_map):
        if elements_orbital_map is None:
            return None
        if not elements_orbital_map:
            return []
        if all(not orbs for orbs in elements_orbital_map.values()):
            return []
        #
        all_orbs_list = elements_orbital_map.values()
        max_orb_l = max([orb_l for orbs in all_orbs_list for orb_l in orbs])
        orbital_counts = np.zeros(max_orb_l + 1, dtype=int)
        for orbital_type in elements_orbital_map.values():
            for orb_l in range(max_orb_l+1):
                orbital_counts[orb_l] = max(orbital_counts[orb_l], orbital_type.count(orb_l))
        common_orbital_types = []
        for orb_l in range(max_orb_l + 1):
            common_orbital_types.extend([orb_l] * orbital_counts[orb_l])
        return common_orbital_types

    @staticmethod
    def _get_common_num(common_orbital_types):
        if not common_orbital_types:
            return 0
        return sum(map(lambda x: 2*x+1, common_orbital_types))
