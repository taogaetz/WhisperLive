{
  config,
  lib,
  ...
}:

let
  cfg = config.services.whisperlivePascal;
in
{
  options.services.whisperlivePascal = {
    enable = lib.mkEnableOption "WhisperLive tuned for NVIDIA Pascal GPUs";

    image = lib.mkOption {
      type = lib.types.str;
      default = "ghcr.io/taogaetz/whisperlive@sha256:5d039b2a6ef2dd976b1ac112549263c45656ff22832a1375cfab8ed58977f9ca";
      description = "OCI image reference pinned to a published Pascal image.";
    };

    listenAddress = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address on which the host publishes both APIs.";
    };

    websocketPort = lib.mkOption {
      type = lib.types.port;
      default = 9090;
      description = "Host port for live WebSocket transcription.";
    };

    restPort = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Host port for the OpenAI-compatible REST API.";
    };

    stateDirectory = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/whisperlive";
      description = "Persistent model and Hugging Face cache directory.";
    };

    computeType = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "float32";
      description = "Optional CTranslate2 precision override.";
    };
  };

  config = lib.mkIf cfg.enable {
    virtualisation.docker.enable = true;
    virtualisation.oci-containers.backend = "docker";
    virtualisation.oci-containers.containers.whisperlive-pascal = {
      image = cfg.image;
      autoStart = true;
      ports = [
        "${cfg.listenAddress}:${toString cfg.websocketPort}:9090"
        "${cfg.listenAddress}:${toString cfg.restPort}:8000"
      ];
      volumes = [ "${cfg.stateDirectory}:/models" ];
      environment = lib.optionalAttrs (cfg.computeType != null) {
        WHISPERLIVE_COMPUTE_TYPE = cfg.computeType;
      };
      extraOptions = [ "--device=nvidia.com/gpu=all" ];
    };

    systemd.tmpfiles.rules = [
      "d ${cfg.stateDirectory} 0750 10001 10001 -"
    ];
  };
}
