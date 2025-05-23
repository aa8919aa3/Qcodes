import json

import numpy
import pytest

from qcodes.dataset.measurements import Measurement
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.parameters import ManualParameter
from qcodes.station import Station


@pytest.fixture  # scope is "function" per default
def dac():
    dac = DummyInstrument("dummy_dac", gates=["ch1", "ch2"])
    yield dac
    dac.close()


@pytest.fixture
def dmm():
    dmm = DummyInstrument("dummy_dmm", gates=["v1", "v2"])
    yield dmm
    dmm.close()


@pytest.mark.parametrize("pass_station", (True, False))
def test_station_snapshot_during_measurement(
    experiment, dac, dmm, pass_station
) -> None:
    station = Station()
    station.add_component(dac)
    station.add_component(dmm, "renamed_dmm")

    snapshot_of_station = station.snapshot()

    if pass_station:
        measurement = Measurement(experiment, station)
    else:
        # in this branch of the `if` we expect that `Measurement` object
        # will be initialized with `Station.default` which is equal to the
        # station object that is instantiated above
        measurement = Measurement(experiment)

    measurement.register_parameter(dac.ch1)
    measurement.register_parameter(dmm.v1, setpoints=[dac.ch1])
    snapshot_of_parameters = {
        parameter.short_name: parameter.snapshot() for parameter in (dac.ch1, dmm.v1)
    }
    snapshot_of_parameters.update(
        {
            parameter.register_name: parameter.snapshot()
            for parameter in (dac.ch1, dmm.v1)
        }
    )
    with measurement.run() as data_saver:
        data_saver.add_result((dac.ch1, 7), (dmm.v1, 5))

    # 1. Test `get_metadata('snapshot')` method
    # this is not part of the DatasetProtocol interface
    # but we test it anyway
    json_snapshot_from_dataset = data_saver.dataset.get_metadata("snapshot")  # type: ignore[attr-defined]
    snapshot_from_dataset = json.loads(json_snapshot_from_dataset)

    expected_snapshot = {
        "station": snapshot_of_station,
        "parameters": snapshot_of_parameters,
    }
    assert expected_snapshot == snapshot_from_dataset

    # 2. Test `snapshot_raw` property
    # this is not part of the DatasetProtocol interface
    # but we test it anyway
    assert json_snapshot_from_dataset == data_saver.dataset.snapshot_raw  # type: ignore[attr-defined]

    # 3. Test `snapshot` property

    assert expected_snapshot == data_saver.dataset.snapshot


def test_snapshot_creation_for_types_not_supported_by_builtin_json(experiment) -> None:
    """
    Test that `Measurement`/`Runner`/`DataSaver` infrastructure
    successfully dumps station snapshots in JSON format in cases when the
    snapshot contains data of types that are not supported by python builtin
    `json` module, for example, numpy scalars.
    """
    p1 = ManualParameter("p_np_int32", initial_value=numpy.int32(5))
    p2 = ManualParameter("p_np_float16", initial_value=numpy.float16(5.0))
    p3 = ManualParameter("p_np_array", initial_value=numpy.meshgrid((1, 2), (3, 4)))
    p4 = ManualParameter("p_np_bool", initial_value=numpy.bool_(False))

    station = Station(p1, p2, p3, p4)

    measurement = Measurement(experiment, station)

    # we need at least 1 parameter to be able to run the measurement
    measurement.register_custom_parameter("dummy")

    with measurement.run() as data_saver:
        # we do this in order to create a snapshot of the station and add it
        # to the database
        pass

    snapshot = data_saver.dataset.snapshot
    assert snapshot is not None

    assert 5 == snapshot["station"]["parameters"]["p_np_int32"]["value"]
    assert 5 == snapshot["station"]["parameters"]["p_np_int32"]["raw_value"]

    assert 5.0 == snapshot["station"]["parameters"]["p_np_float16"]["value"]
    assert 5.0 == snapshot["station"]["parameters"]["p_np_float16"]["raw_value"]

    lst = [[[1, 2], [1, 2]], [[3, 3], [4, 4]]]
    assert lst == snapshot["station"]["parameters"]["p_np_array"]["value"]
    assert lst == snapshot["station"]["parameters"]["p_np_array"]["raw_value"]

    assert False is snapshot["station"]["parameters"]["p_np_bool"]["value"]
    assert False is snapshot["station"]["parameters"]["p_np_bool"]["raw_value"]
