{
  config,
  lib,
  ...
}:

let
  cfg = config.services.pascalScribe;
in
{
  options.services.pascalScribe = {
    enable = lib.mkEnableOption "PascalScribe";

    image = lib.mkOption {
      type = lib.types.str;
      default = "ghcr.io/taogaetz/pascalscribe:1080ti";
      description = "OCI image reference for the published 1080 Ti image.";
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

    historyDirectory = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/pascalscribe-history";
      description = "Persistent transcript history directory.";
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
    virtualisation.oci-containers.containers.pascalscribe = {
      image = cfg.image;
      autoStart = true;
      ports = [
        "${cfg.listenAddress}:${toString cfg.websocketPort}:9090"
        "${cfg.listenAddress}:${toString cfg.restPort}:8000"
      ];
      volumes = [ "${cfg.historyDirectory}:/data" ];
      environment = lib.optionalAttrs (cfg.computeType != null) {
        PASCALSCRIBE_COMPUTE_TYPE = cfg.computeType;
      };
      extraOptions = [ "--device=nvidia.com/gpu=all" ];
    };

    systemd.tmpfiles.rules = [
      "d ${cfg.historyDirectory} 0750 10001 10001 -"
    ];
  };
}
