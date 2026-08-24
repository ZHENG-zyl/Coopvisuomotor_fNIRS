# Coopvisuomotor-fNIRS
Cooperative visuomotor control requires partners to coordinate complementary actions while continuously monitoring a shared outcome. However, open hyperscanning datasets that combine interacting-brain recordings with synchronized kinematic and force measurements during cooperative visuomotor control remain scarce. Here, we present a functional near-infrared spectroscopy (fNIRS) hyperscanning dataset collected from 64 healthy adults organized into 32 dyads while they performed a virtual coloring task using two haptic-feedback devices. Each dyad completed the task under two collaboration modes and two difficulty levels. In the leader-follower mode, one participant controlled the pen trajectory and the other controlled contact force, which determined brush width; in the egalitarian mode, the participants controlled orthogonal position components and jointly determined brush width. Task difficulty was manipulated by target-pattern complexity. Each participant was recorded with 26-channel fNIRS covering the left and right prefrontal cortex, left sensorimotor cortex, and right temporoparietal junction. Synchronized behavioral data, including continuous pen kinematics, contact force signals, event markers, and trial-wise collaboration performance metrics, were acquired in parallel. The dataset contains fNIRS recordings, metadata, event annotations, behavioral files, and example analysis code. Technical validation includes behavioral validation of task difficulty, fNIRS signal-quality assessment, task-evoked hemoglobin responses, and example analyses of inter-brain synchrony. This dataset provides a resource for studying joint action, neural-behavioral coupling, role-dependent coordination, and computational methods for multimodal hyperscanning data.

This dataset contains simultaneous fNIRS hyperscanning and haptic-device behavioral data from 64 healthy adults organized into 32 dyads. Each dyad completed 15 valid trials in each of four fixed-order conditions: leader-follower/carrot (`run-01`), leader-follower/wave (`run-02`), egalitarian/carrot (`run-03`), and egalitarian/wave (`run-04`).

## Organization

The dataset uses a SNIRF-based, BIDS-inspired dyad-level layout. It is not presented as strict BIDS because one fNIRS system stored both participants in a single 52-channel recording and the behavioral data are also dyad-level. Channels 1-26 belong to member A and channels 27-52 to member B. Participant identifiers use the form `dyad-001A` and `dyad-001B`.

- `dyad-[001-032]/nirs/`: four validated SNIRF recordings per dyad.
- `dyad-[001-032]/beh/`: continuous behavioral samples and trial-level summaries for each run.
- `sourcedata/`: native unprocessed `.nirs` recordings corresponding to the released SNIRF files.
- `derivatives/technical-validation/`: machine-readable results reported in the Technical Validation section.
- `code/`: final scripts and parameter files for the three technical-validation analyses.
- `stimuli/`: reference images of the task interfaces.

The aggregate `task-coopvm_events.tsv` contains 1,920 trial onsets (32 dyads x 4 runs x 15 trials). Onsets are derived from the fNIRS trial-start markers, event duration is the fixed 20-s coloring period, and `completion_time_s` records the behavioral completion time. `task-coopvm_channel-pairs.tsv` defines the 26 cross-participant channel pairs used in the WTC analysis.

## fNIRS acquisition

The NirScan continuous-wave system recorded 52 dyadic channels at 11 Hz using wavelengths of 730, 808, and 850 nm. All 128 SNIRF files passed the official `pysnirf2` 0.7.3 validator on 2026-08-04. Source-detector geometry is shared across all released recordings.

Intermediate Homer files used for the QC, HRF, and WTC analyses are not distributed. They can be regenerated from `sourcedata/` using the workflows recorded in `code/preprocessing_hrf_parameters.json` and `code/preprocessing_wtc_parameters.json`.

## Behavioral data

Continuous behavioral files contain trial-indexed samples from both Touch haptic devices at 1000 Hz. They do not contain an absolute clock column; within-trial elapsed time can be reconstructed as the zero-based row index divided by 1000. Trial summaries provide force, position, total scores, and actual completion time.

## Reuse notes

Run order was fixed, so condition effects may be confounded with order or fatigue. Roles were fixed by member label within each collaboration mode. The montage did not include short-separation channels. Broad-region labels describe montage intent, whereas the MNI coordinates and anatomical labels in the channel table report the AtlasViewer output.

## License and citation

Data and documentation are released under CC BY 4.0. Cite the accompanying data descriptor and the dataset DOI: https://doi.org/10.57760/sciencedb.46254.
