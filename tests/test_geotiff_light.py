import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

_RASTERIO_AVAILABLE = importlib.util.find_spec("rasterio") is not None


class TestGeoTiffLight(unittest.TestCase):
    def test_missing_rasterio_raises_geo_tiff_error(self):
        from src.data.light import GeoTiffError, sample_light_kW_m2

        if _RASTERIO_AVAILABLE:
            self.skipTest("rasterio is installed in this environment")

        with self.assertRaises(GeoTiffError):
            sample_light_kW_m2(
                Path("dummy.tif"),
                lat=0.0,
                lon=0.0,
            )

    @unittest.skipUnless(_RASTERIO_AVAILABLE, "requires rasterio")
    def test_samples_value_from_epsg4326_geotiff(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        from src.data.light import GeoTiffSampler

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tif = tmp / "light.tif"

            data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
            transform = from_origin(0.0, 2.0, 1.0, 1.0)

            with rasterio.open(
                tif,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
                nodata=-9999.0,
            ) as dst:
                dst.write(data, 1)

            with GeoTiffSampler(tif, band=1) as sampler:
                v1 = sampler.sample_lat_lon(lat=1.5, lon=0.5)
                v2 = sampler.sample_lat_lon(lat=0.5, lon=1.5)
                v3 = sampler.sample_lat_lon(lat=10.0, lon=10.0)

            self.assertTrue(math.isclose(v1 or 0.0, 10.0, rel_tol=0.0, abs_tol=1e-6))
            self.assertTrue(math.isclose(v2 or 0.0, 40.0, rel_tol=0.0, abs_tol=1e-6))
            self.assertIsNone(v3)


if __name__ == "__main__":
    unittest.main()
