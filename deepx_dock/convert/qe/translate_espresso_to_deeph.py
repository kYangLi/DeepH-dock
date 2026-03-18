from pathlib import Path
from functools import partial

from deepx_dock.parallel import parallel_map
from deepx_dock.hpro.kernel import PW2AOkernel
from deepx_dock.misc import get_data_dir_lister

QE_VSC_FILENAME = "VSC"
QE_OUT_FILENAME = "scf.out"
QE_RHO_FILENAME = "charge_density.h5"
QE_NECESSARY_FILES = set([QE_VSC_FILENAME, QE_OUT_FILENAME])


def validation_check_qe(root_dir: Path, prev_dirname: Path):
    all_files = [str(v.name) for v in root_dir.iterdir()]
    if QE_NECESSARY_FILES.issubset(set(all_files)):
        yield prev_dirname


class EspressoDatasetTranslator:
    def __init__(self,
        espresso_data_dir, deeph_data_dir, basis_dir, 
        export_S=True, export_H=True, export_rho=False, export_r=False,
        n_jobs=1, n_tier=0
    ):
        self.espresso_data_dir = Path(espresso_data_dir)
        self.deeph_data_dir = Path(deeph_data_dir)
        self.basis_dir = Path(basis_dir)
        self.export_S = export_S
        self.export_H = export_H
        self.export_rho = export_rho
        self.export_r = export_r
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        self.deeph_data_dir.mkdir(parents=True, exist_ok=True)

    def transfer_all_espresso_to_deeph(self):
        worker = partial(
            self.transfer_one_espresso_to_deeph,
            espresso_path=self.espresso_data_dir,
            basis_path=self.basis_dir,
            deeph_path=self.deeph_data_dir,
            export_S=self.export_S,
            export_H=self.export_H,
            export_rho=self.export_rho,
            export_r=self.export_r,
        )
        data_dir_lister = get_data_dir_lister(
            self.espresso_data_dir, self.n_tier, validation_check_qe
        )
        parallel_map(worker, data_dir_lister, n_jobs=self.n_jobs, desc="Data")

    @staticmethod
    def transfer_one_espresso_to_deeph(
        dir_name: str, espresso_path: Path, basis_path: Path, deeph_path: Path, 
        export_S=True, export_H=True, export_rho=False, export_r=False
    ):
        espresso_dir_path = espresso_path / dir_name
        if not espresso_dir_path.is_dir():
            return
        epsresso_vsc_dir = espresso_dir_path / QE_VSC_FILENAME
        deeph_dir_path = deeph_path / dir_name
        deeph_dir_path.mkdir(parents=True, exist_ok=True)
        #
        kernel = PW2AOkernel(
            aodata_interface='siesta',
            aodata_root=basis_path,
            hrdata_interface='qe-bgw',
            vscdir=epsresso_vsc_dir,
            upfdir=basis_path,
            ecutwfn=50.0,
        )
        kernel.run_pw2ao_rs(deeph_dir_path)