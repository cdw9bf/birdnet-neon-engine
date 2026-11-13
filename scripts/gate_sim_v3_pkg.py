"""Detection replay for BirdNET v3.0 packages (the V002 and K2B-G1 gates;
also the global G003 form).

The top-10 caches written by gate_v3_pkg.py (both sides, global label
indices) go through the v3 harness (simulate5s.py, imported read-only:
BirdNET-Go 20261014 post-processing, v3_global form = top-10 over all
labels) at overlap 2.2 / threshold 0.84 / FP level 3 with the v3 geomodel
range filter at 0.05, at two locations: the evaluation site given with
--site (each stream's own week) and each stream's own. Paired bootstrap over
periods (2000 reps). Report-only: the same with the evaluation server's
existing reference cache (scores_v3 g_idx/g_conf) as the reference.

usage: python -I gate_sim_v3_pkg.py <tuning_dir> <eval_dir> <scores_dir>
           --site LAT,LON [label] [--regional] [--ref-name NAME]
  --site:     the evaluation site of the range filter (required; the
              published results used one site in the eastern US, see
              docs/GATES.md)
  --regional: the model is the NA-east slice; the report-only cache
              reference is the cache's 'reg' (800-class) scores
  --ref-name: the reference's name in the report (default "fp32 ORT (this
              run)"; the global G003 form: "fp32 ORT global (this run)")
"""
import sys

from bnneon import replay_v3

REF_NAME = "fp32 ORT (this run)"


def parse(argv: list[str]
          ) -> tuple[list[str], bool, str, replay_v3.Site]:
    """(positional arguments, --regional, --ref-name, --site) of the command
    line.

    Raises:
        ValueError: --site is missing or malformed.
    """
    args = list(argv[1:])
    site = replay_v3.take_site(args)
    ref_name = REF_NAME
    if "--ref-name" in args:
        i = args.index("--ref-name")
        ref_name = args[i + 1]
        del args[i:i + 2]
    regional = "--regional" in args
    return [a for a in args if a != "--regional"], regional, ref_name, site


def main(argv: list[str]) -> int:
    """Prints the replay report; returns 0, or 2 for a bad command line."""
    try:
        args, regional, ref_name, site = parse(argv)
    except ValueError as e:
        print(f"gate_sim_v3_pkg: {e}", file=sys.stderr)
        return 2
    if len(args) < 3:
        print(__doc__[__doc__.index("usage:"):], file=sys.stderr)
        return 2
    tuning, ev, sc = args[0], args[1], args[2]
    label = args[3] if len(args) > 3 else "neon"
    sim = replay_v3.load_harness(tuning, ev)
    geo_site = replay_v3.site_geo(sim, site)
    scores = replay_v3.load_scores(sim, sc, regional)
    replay_v3.report(sim, site, geo_site, scores, ref_name, label)
    replay_v3.coverage(sim, geo_site)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
