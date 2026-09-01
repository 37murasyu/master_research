import glob
import os
import numpy as np
import pandas as pd

FILES = glob.glob('output_data/cycle_energy/*gcvspl*cycle_work_shoulder.csv')
TARGET_PARTS = {'elbow_R', 'wrist_R'}
LOWER, UPPER = 0.5, 1.5
WINSOR_Q = (0.05, 0.95)  # robustly clamp per subject/part
OUT_DIR = 'output_data/cycle_energy'
SUBJECT_OUT = os.path.join(OUT_DIR, 'ratio_pos_vs_one_rm_subject_stats.csv')
PART_OUT = os.path.join(OUT_DIR, 'ratio_pos_vs_one_rm_part_stats.csv')
ALIGN_OUT = os.path.join(OUT_DIR, 'ratio_pos_vs_one_rm_elbow_wrist_alignment.csv')


def main():
    if not FILES:
        raise SystemExit('no gcvspl cycle_work_shoulder files found')

    frames = []
    for path in FILES:
        df = pd.read_csv(path)
        df = df[df['part'].isin(TARGET_PARTS)].copy()
        frames.append(df[['subject', 'part', 'cycle_index', 'ratio_pos_vs_one_rm']])
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(['subject', 'part', 'cycle_index'])

    # drop the first entry per subject-part as outlier
    all_df['idx'] = all_df.groupby(['subject', 'part']).cumcount()
    all_df = all_df[all_df['idx'] > 0].drop(columns='idx')

    if all_df.empty:
        raise SystemExit('no rows left after dropping first cycles')

    # winsorize per subject/part to clamp extremes (after dropping first cycle)
    def _winsor(series: pd.Series) -> pd.Series:
        lo, hi = series.quantile(WINSOR_Q[0]), series.quantile(WINSOR_Q[1])
        return series.clip(lo, hi)

    all_df['ratio_pos_vs_one_rm'] = (
        all_df.groupby(['subject', 'part'])['ratio_pos_vs_one_rm']
        .transform(_winsor)
    )

    # subject-part stats on winsorized values
    grouped = all_df.groupby(['subject', 'part'])['ratio_pos_vs_one_rm']
    med = grouped.median()
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = q3 - q1
    cover = grouped.apply(lambda s: ((LOWER <= s) & (s <= UPPER)).mean())
    n = grouped.size()

    subj_stats = (
        pd.DataFrame({
            'median': med,
            'iqr': iqr,
            'coverage': cover,
            'n_cycles': n,
        })
        .reset_index()
        .sort_values(['part', 'subject'])
    )

    # part-level robust summaries
    part_results = []
    for part, subdf in subj_stats.groupby('part'):
        medians = subdf['median'].values
        iqrs = subdf['iqr'].values
        m_tilde = np.median(medians)
        mad = np.median(np.abs(medians - m_tilde))
        rcv = (1.4826 * mad / m_tilde) if m_tilde != 0 else np.nan
        iqr_med = np.median(iqrs)
        iqr_p90 = np.percentile(iqrs, 90)

        part_df = all_df[all_df['part'] == part]
        part_cover = ((LOWER <= part_df['ratio_pos_vs_one_rm']) & (part_df['ratio_pos_vs_one_rm'] <= UPPER)).mean()

        part_results.append({
            'part': part,
            'median_of_medians': m_tilde,
            'mad_of_medians': mad,
            'robust_cv': rcv,
            'iqr_median': iqr_med,
            'iqr_p90': iqr_p90,
            'overall_coverage': part_cover,
            'n_subjects': len(subdf),
            'n_cycles_total': len(part_df),
        })

    part_stats = pd.DataFrame(part_results).sort_values('part')

    # elbow-wrist alignment within subject (same cycle_index)
    align_rows = []
    elbow_df = all_df[all_df['part'] == 'elbow_R']
    wrist_df = all_df[all_df['part'] == 'wrist_R']
    merged = pd.merge(
        elbow_df,
        wrist_df,
        on=['subject', 'cycle_index'],
        suffixes=('_elbow', '_wrist'),
    )

    for subject, subdf in merged.groupby('subject'):
        e = subdf['ratio_pos_vs_one_rm_elbow']
        w = subdf['ratio_pos_vs_one_rm_wrist']
        diff = w - e
        med_diff = diff.median()
        mad_diff = np.median(np.abs(diff - med_diff))
        base = np.median((w + e) / 2)
        rel_mad = 1.4826 * mad_diff / base if base != 0 else np.nan
        rho = subdf[['ratio_pos_vs_one_rm_elbow', 'ratio_pos_vs_one_rm_wrist']].corr(method='spearman').iloc[0, 1] if len(subdf) >= 2 else np.nan
        dual_cover = (((LOWER <= w) & (w <= UPPER) & (LOWER <= e) & (e <= UPPER))).mean()
        sign_agree = (np.sign(w - 1) == np.sign(e - 1)).mean()

        align_rows.append({
            'subject': subject,
            'n_pairs': len(subdf),
            'spearman_rho': rho,
            'median_diff_w_minus_e': med_diff,
            'mad_diff': mad_diff,
            'rel_mad_diff': rel_mad,
            'dual_coverage': dual_cover,
            'sign_agreement': sign_agree,
        })

    align_stats = pd.DataFrame(align_rows).sort_values('subject') if align_rows else pd.DataFrame()

    os.makedirs(OUT_DIR, exist_ok=True)
    subj_stats.to_csv(SUBJECT_OUT, index=False)
    part_stats.to_csv(PART_OUT, index=False)
    if not align_stats.empty:
        align_stats.to_csv(ALIGN_OUT, index=False)

    print('\n[Subject x Part stats (first cycles dropped)]')
    print(subj_stats)
    print('\n[Part-level robust summaries]')
    print(part_stats)
    if not align_stats.empty:
        print('\n[Elbow vs Wrist alignment per subject]')
        print(align_stats)
    print('\nSaved:')
    print(' ', os.path.abspath(SUBJECT_OUT))
    print(' ', os.path.abspath(PART_OUT))
    if not align_stats.empty:
        print(' ', os.path.abspath(ALIGN_OUT))


if __name__ == '__main__':
    main()
