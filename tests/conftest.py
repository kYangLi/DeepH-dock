import pytest
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
TESTS_DIR = ROOT_DIR / "tests"
EXAMPLES_DIR = ROOT_DIR / "examples"


@pytest.fixture
def tests_dir():
    """Provide tests directory"""
    return TESTS_DIR


@pytest.fixture
def examples_dir():
    """Provide examples directory"""
    return EXAMPLES_DIR


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary output directory (under /tmp)"""
    return tmp_path


@pytest.fixture
def siesta_test_data(tests_dir):
    """SIESTA test data fixture"""
    return {
        "input": tests_dir / "convert/siesta/siesta.bak",
        "reference": tests_dir / "convert/siesta/deeph.bak",
    }


@pytest.fixture
def openmx_test_data(tests_dir):
    """OpenMX test data fixture"""
    return {
        "input": tests_dir / "convert/openmx/openmx.bak",
        "reference": tests_dir / "convert/openmx/deeph.bak",
    }


@pytest.fixture
def abacus_test_data(tests_dir):
    """ABACUS test data fixture"""
    return {
        "input": tests_dir / "convert/abacus/abacus.bak",
        "reference": tests_dir / "convert/abacus/deeph.bak",
    }


@pytest.fixture
def fhi_aims_test_data(tests_dir):
    """FHI-aims test data fixture"""
    return {
        "input": tests_dir / "convert/fhi_aims/single_atoms_aims.bak",
        "reference": tests_dir / "convert/fhi_aims/single_atoms_deeph.bak",
    }


@pytest.fixture
def hopcp_test_data(tests_dir):
    """HOPCP test data fixture"""
    return {
        "input": tests_dir / "convert/hopcp/petsc.bak",
        "reference": tests_dir / "convert/hopcp/deeph.bak",
    }


@pytest.fixture
def deeph_test_data(tests_dir):
    """DeepH format test data fixture"""
    return {
        "input": tests_dir / "convert/deeph/legacy.bak",
        "reference": tests_dir / "convert/deeph/standardize.bak",
    }


@pytest.fixture
def eigen_test_data(tests_dir):
    """Eigenvalue calculation test data fixture"""
    return {
        "input": tests_dir / "compute/eigen/eigen.clean",
        "reference": tests_dir / "compute/eigen/eigen.bak",
    }


@pytest.fixture
def error_analysis_test_data(tests_dir):
    """Error analysis test data fixture"""
    return {
        "benchmark": tests_dir / "analyze/error/benchmark.bak",
        "input": tests_dir / "analyze/error/infer.clean",
        "reference": tests_dir / "analyze/error/infer.bak",
    }


@pytest.fixture
def dataset_test_data(tests_dir):
    """Dataset analysis test data fixture"""
    return {
        "input": tests_dir / "analyze/dataset/inputs.clean",
        "reference": tests_dir / "analyze/dataset/inputs.bak",
    }


@pytest.fixture
def dft_equiv_test_data(tests_dir):
    """DFT equivariance test data fixture"""
    return {
        "input": tests_dir / "analyze/dft_equiv/poscars.clean",
        "reference": tests_dir / "analyze/dft_equiv/dft_calc.bak",
    }
