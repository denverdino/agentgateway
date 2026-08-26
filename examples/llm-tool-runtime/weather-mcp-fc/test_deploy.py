from __future__ import annotations

import importlib.util
import os
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("deploy.py")


def load_module():
    spec = importlib.util.spec_from_file_location("weather_mcp_fc_deploy", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load deployment module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.deploy = load_module()

    def test_load_configuration_reads_only_allowlisted_dotenv_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dotenv = Path(temporary_directory) / ".env"
            dotenv.write_text(
                "ALIBABA_CLOUD_ACCESS_KEY_ID=dotenv-id\n"
                "ALIBABA_CLOUD_ACCESS_KEY_SECRET=dotenv-secret\n"
                "ALIBABA_CLOUD_REGION=cn-hangzhou\n"
                "WEATHERAPI_KEY=weather-key\n"
                "FC_WEATHER_MCP_FUNCTION_NAME=custom-weather-mcp\n"
                "UNRELATED_SECRET=do-not-read\n",
                encoding="utf-8",
            )

            settings = self.deploy.DeploymentSettings.load(
                dotenv, {"ALIBABA_CLOUD_ACCESS_KEY_ID": "environment-id"}
            )

            self.assertEqual(settings.access_key_id, "environment-id")
            self.assertEqual(settings.access_key_secret, "dotenv-secret")
            self.assertEqual(settings.region, "cn-hangzhou")
            self.assertEqual(settings.function_name, "custom-weather-mcp")
            self.assertEqual(settings.trigger_name, "weatherapi-mcp-http")
            self.assertNotIn("do-not-read", repr(settings))

    def test_bootstrap_selects_the_fc_custom_runtime_node20_binary(self) -> None:
        bootstrap = MODULE_PATH.with_name("bootstrap").read_text(encoding="utf-8")

        self.assertIn("exec /var/fc/lang/nodejs20/bin/node server.mjs", bootstrap)

    def test_missing_configuration_reports_names_without_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dotenv = Path(temporary_directory) / ".env"
            dotenv.write_text("WEATHERAPI_KEY=must-not-leak\n", encoding="utf-8")

            with self.assertRaises(self.deploy.DeploymentError) as raised:
                self.deploy.DeploymentSettings.load(dotenv, {})

            message = str(raised.exception)
            self.assertIn("ALIBABA_CLOUD_ACCESS_KEY_ID", message)
            self.assertIn("ALIBABA_CLOUD_ACCESS_KEY_SECRET", message)
            self.assertIn("ALIBABA_CLOUD_REGION", message)
            self.assertNotIn("must-not-leak", message)

    def test_function_and_trigger_inputs_describe_a_custom_runtime_web_function(self) -> None:
        settings = self.deploy.DeploymentSettings(
            access_key_id="id",
            access_key_secret="secret",
            region="cn-hangzhou",
            weatherapi_key="weather-key",
            ingress_token="ingress-token",
            function_name="agentgateway-weatherapi-mcp",
            trigger_name="weatherapi-mcp-http",
        )

        function_input = self.deploy.function_input(settings, "base64-zip")
        trigger_input = self.deploy.trigger_input(settings)

        self.assertEqual(function_input["runtime"], "custom.debian10")
        self.assertEqual(function_input["handler"], "index.handler")
        self.assertEqual(function_input["customRuntimeConfig"], {
            "command": ["./bootstrap"],
            "port": 9000,
        })
        self.assertEqual(function_input["environmentVariables"], {
            "WEATHERAPI_KEY": "weather-key",
            "INGRESS_BEARER_TOKEN": "ingress-token",
        })
        self.assertEqual(function_input["code"], {"zipFile": "base64-zip"})
        self.assertEqual(trigger_input["triggerType"], "http")
        self.assertEqual(trigger_input["triggerConfig"]["authType"], "anonymous")
        self.assertEqual(trigger_input["triggerConfig"]["methods"], ["GET", "POST"])

    def test_build_archive_is_deterministic_and_excludes_development_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "bootstrap").write_text("#!/bin/sh\nexec node server.mjs\n", encoding="utf-8")
            (source / "server.mjs").write_text("export {};\n", encoding="utf-8")
            (source / "package.json").write_text("{}\n", encoding="utf-8")
            (source / "package-lock.json").write_text("{}\n", encoding="utf-8")
            (source / "server.test.mjs").write_text("secret test fixture", encoding="utf-8")
            (source / "deploy.py").write_text("secret deployment source", encoding="utf-8")
            dependency = source / "node_modules" / "dependency"
            dependency.mkdir(parents=True)
            (dependency / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")
            first = root / "first.zip"
            second = root / "second.zip"

            self.deploy.build_archive(source, first)
            os.utime(source / "server.mjs", (1_900_000_000, 1_900_000_000))
            self.deploy.build_archive(source, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "bootstrap",
                        "node_modules/dependency/index.js",
                        "package-lock.json",
                        "package.json",
                        "server.mjs",
                    ],
                )
                bootstrap_mode = archive.getinfo("bootstrap").external_attr >> 16
                self.assertTrue(bootstrap_mode & stat.S_IXUSR)

    def test_normalize_public_url_appends_mcp_path(self) -> None:
        self.assertEqual(
            self.deploy.normalize_public_url("https://example.cn-hangzhou.fcapp.run/"),
            "https://example.cn-hangzhou.fcapp.run/mcp",
        )
        with self.assertRaises(self.deploy.DeploymentError):
            self.deploy.normalize_public_url("http://example.test")

    def test_default_fc_endpoint_is_left_to_the_regional_sdk_map(self) -> None:
        settings = self.deploy.DeploymentSettings(
            access_key_id="id",
            access_key_secret="secret",
            region="cn-hangzhou",
            weatherapi_key="weather-key",
            ingress_token="ingress-token",
            function_name="agentgateway-weatherapi-mcp",
            trigger_name="weatherapi-mcp-http",
        )

        self.assertIsNone(self.deploy.configured_fc_endpoint(settings))

    def test_fc_client_configuration_allows_large_zip_uploads(self) -> None:
        settings = self.deploy.DeploymentSettings(
            access_key_id="id",
            access_key_secret="secret",
            region="cn-hangzhou",
            weatherapi_key="weather-key",
            ingress_token="ingress-token",
            function_name="agentgateway-weatherapi-mcp",
            trigger_name="weatherapi-mcp-http",
        )

        class FakeConfig:
            def __init__(self, **values):
                self.__dict__.update(values)

        configuration = self.deploy.fc_client_configuration(settings, FakeConfig)

        self.assertEqual(configuration.connect_timeout, 10_000)
        self.assertEqual(configuration.read_timeout, 120_000)
        self.assertEqual(configuration.region_id, "cn-hangzhou")
        self.assertFalse(hasattr(configuration, "endpoint"))

    def test_update_dotenv_replaces_managed_values_without_touching_other_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dotenv = Path(temporary_directory) / ".env"
            dotenv.write_text(
                "OPENAI_API_KEY=keep-this\n"
                "FC_WEATHER_MCP_URL=https://old.example/mcp\n"
                "FC_WEATHER_MCP_TOKEN=old-token\n",
                encoding="utf-8",
            )

            self.deploy.update_dotenv(
                dotenv,
                "https://new.example/mcp",
                "new-token",
            )

            contents = dotenv.read_text(encoding="utf-8")
            self.assertIn("OPENAI_API_KEY=keep-this\n", contents)
            self.assertIn("FC_WEATHER_MCP_URL=https://new.example/mcp\n", contents)
            self.assertIn("FC_WEATHER_MCP_TOKEN=new-token\n", contents)
            self.assertNotIn("old-token", contents)
            self.assertEqual(stat.S_IMODE(dotenv.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
