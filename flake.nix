{
  description = "WhisperLive development tools and Pascal NixOS module";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          docker_29
          gitMinimal
          just
          ruff
          shellcheck
          uv
        ];

        shellHook = ''
          export DOCKER_BUILDKIT=1
          echo "WhisperLive Pascal shell: just test | just build | just run"
        '';
      };

      formatter.${system} = pkgs.nixfmt-tree;
      nixosModules.default = import ./nix/module.nix;
    };
}
