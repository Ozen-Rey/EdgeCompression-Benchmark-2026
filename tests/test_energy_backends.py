from src.utils.energy_backends import CompositeEnergyMeter, EnergyReading


def test_composite_energy_meter_returns_provenance():
    meter = CompositeEnergyMeter()

    def work():
        return 123

    result, reading = meter.measure_callable(work)

    assert result == 123
    assert isinstance(reading, EnergyReading)
    assert reading.time_s >= 0.0
    assert isinstance(reading.energy_backend, str)
    assert isinstance(reading.energy_method, str)
    assert isinstance(reading.energy_is_measured, bool)
    assert isinstance(reading.energy_quality, str)
    assert isinstance(reading.warnings, list)
