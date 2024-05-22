{
  description = "Development environment for Python 3.11 scripts and packages";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.2311.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems =
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f:
        nixpkgs.lib.genAttrs supportedSystems
          (system: f { pkgs = import nixpkgs { inherit system; }; });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          name = "python 3.11 development shell";
          packages = with pkgs;
            [
              python311
              poetry
            ] ++ (with pkgs.python311Packages; [
              pip
              icecream
              black
              jupyterlab
              notebook
              nodejs
              ipython
              pandas
              seaborn
              matplotlib
              scipy
              scikit-learn 
            ]);
          shellHook = ''
            echo
            echo "Activated environment for Python 3.11 development"
          '';
        };
      });
    };
}
