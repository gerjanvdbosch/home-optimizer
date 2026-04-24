# TASKLIST.md — Home Optimizer professioneel implementatieplan

## Fase 0 — Projectbasis

### 0.1 Repository-structuur
- [ ] Maak package-structuur:
  - `home_optimizer/config`
  - `home_optimizer/models`
  - `home_optimizer/discretization`
  - `home_optimizer/estimation`
  - `home_optimizer/forecasting`
  - `home_optimizer/supervision`
  - `home_optimizer/mpc`
  - `home_optimizer/learning`
  - `home_optimizer/diagnostics`
  - `tests`
- [ ] Voeg `pyproject.toml` toe.
- [ ] Voeg linting/typechecking toe:
  - `ruff`
  - `mypy`
  - `pytest`
- [ ] Stel CI-testcommand in:
  - `pytest`
  - `mypy home_optimizer`

---

# Fase 1 — Configuratie en validatie

## 1.1 Configmodellen
- [ ] Implementeer `PhysicalConstantsConfig`.
- [ ] Implementeer `DiscretizationConfig`.
- [ ] Implementeer `UfhConfig`.
- [ ] Implementeer `DhwConfig`.
- [ ] Implementeer `HeatPumpConfig`.
- [ ] Implementeer `EstimatorConfig`.
- [ ] Implementeer `MpcConfig`.
- [ ] Implementeer `SupervisorConfig`.
- [ ] Implementeer `ForecastConfig`.
- [ ] Implementeer `LearningConfig`.
- [ ] Implementeer root-config `HomeOptimizerConfig`.

## 1.2 Harde validatieregels
- [ ] Valideer alle capaciteiten `> 0`.
- [ ] Valideer alle thermische weerstanden `> 0`.
- [ ] Valideer `delta_t > 0`.
- [ ] Valideer `0 <= alpha_solar <= 1`.
- [ ] Valideer `0 <= eta_window <= 1`.
- [ ] Valideer heater split:
  - `0 <= heater_split_top <= 1`
  - `0 <= heater_split_bottom <= 1`
  - som binnen `split_sum_tolerance`
- [ ] Valideer `lambda_water_ref`.
- [ ] Valideer DHW capacity balance.
- [ ] Valideer COP-grenzen.
- [ ] Valideer safety-temperaturen.
- [ ] Valideer covariance matrices:
  - procesruis PSD
  - meetruis PD

## 1.3 Magic-number policy
- [ ] Definieer alle fysische constanten in config.
- [ ] Definieer alle numerieke tolerances in config.
- [ ] Zoek en verwijder losse numerieke constanten uit modelcode, behalve `0` en `1`.

---

# Fase 2 — Thermische model-laag

## 2.1 Generieke state-space klassen
- [ ] Implementeer `ContinuousLinearModel`.
- [ ] Implementeer `DiscreteLinearModel`.
- [ ] Voeg matrixdimensievalidatie toe.
- [ ] Voeg state/input/disturbance metadata toe.
- [ ] Voeg docstrings met eenheden toe.

## 2.2 Discretizer
- [ ] Implementeer `Discretizer`.
- [ ] Implementeer `exact_zoh` via augmented matrix exponential.
- [ ] Implementeer `forward_euler`.
- [ ] Voeg Euler-admissibility interface toe.
- [ ] Voeg tests toe voor dimensies en ZOH-output.

## 2.3 UFH-model
- [ ] Implementeer `UfhContinuousModel`.
- [ ] Bouw `A_c_ufh`.
- [ ] Bouw `B_c_ufh`.
- [ ] Bouw `E_c_ufh`.
- [ ] Implementeer solar gain:
  - `Q_solar = A_glass_eff * GTI_window * eta_window / power_unit_scale`
- [ ] Implementeer UFH energiebalansfunctie.
- [ ] Implementeer UFH Euler-admissibility checks:
  - spectral radius
  - self-damping nonnegative
  - geen tekenomkering
- [ ] Implementeer UFH observeerbaarheidscheck.
- [ ] Voeg tests toe:
  - `test_ufh_energy_balance`
  - `test_ufh_observability_rank`
  - `test_ufh_observability_conditioning`
  - `test_forward_euler_self_damping_nonnegative_ufh`

## 2.4 DHW-model
- [ ] Implementeer `DhwContinuousModel`.
- [ ] Bouw LTV `A_c_dhw[k]`.
- [ ] Bouw `B_c_dhw`.
- [ ] Bouw LTV `E_c_dhw[k]`.
- [ ] Implementeer DHW energiebalansfunctie.
- [ ] Implementeer `T_dhw_energy`.
- [ ] Implementeer `E_dhw_state_rel`.
- [ ] Implementeer DHW Euler-admissibility checks:
  - spectral radius
  - top self-damping
  - bottom self-damping
  - worst-case `Vdot_tap`
- [ ] Implementeer DHW observeerbaarheidscheck.
- [ ] Voeg tests toe:
  - `test_dhw_energy_balance`
  - `test_capacity_balance`
  - `test_dhw_observability_rank`
  - `test_dhw_observability_conditioning`
  - `test_dhw_forward_euler_worst_case_vtap_validation`

## 2.5 Coupled thermal model
- [ ] Implementeer `CoupledThermalModel`.
- [ ] Ondersteun variant A: thermisch ontkoppeld.
- [ ] Ondersteun variant B: indoor-coupled tank losses.
- [ ] Weiger block-diagonal opbouw bij indoor-coupled mode.
- [ ] Weiger dubbele semantiek van `T_amb_tank` als exogeen én state-koppeling.
- [ ] Voeg test toe:
  - `test_block_diagonal_forbidden_when_tank_losses_couple_to_room`

---

# Fase 3 — Estimator-laag

## 3.1 Lineair Kalman-filter
- [ ] Implementeer `LinearKalmanFilter`.
- [ ] Implementeer predictiestap.
- [ ] Implementeer update met Joseph-vorm.
- [ ] Valideer covariance symmetry.
- [ ] Valideer covariance PSD binnen tolerantie.
- [ ] Voeg tests toe:
  - `test_kalman_covariance_psd`

## 3.2 Extended Kalman Filter
- [ ] Implementeer `ExtendedKalmanFilter`.
- [ ] Gebruik discrete transitiecallback `f_d`.
- [ ] Gebruik Jacobiaanprovider die exact bij `f_d` hoort.
- [ ] Implementeer Joseph-update via gedeelde algebra.
- [ ] Implementeer post-update clamp:
  - `Vdot_tap_hat = max(0, Vdot_tap_hat)`
- [ ] Voeg covariance PSD-validatie toe.
- [ ] Voeg lokale observeerbaarheidsdiagnostiek toe:
  - gradient aanwezig
  - zero-gradient case bounded
- [ ] Voeg tests toe:
  - `test_ekf_covariance_psd`
  - `test_ekf_vtap_nonnegative`
  - `test_ekf_jacobian_eval_point`
  - `test_projected_ekf_clamp_is_forwarded_to_mpc`
  - `test_ekf_unobservable_zero_gradient_case_is_bounded`

## 3.3 Estimator facade
- [ ] Implementeer `StateEstimatorFacade`.
- [ ] Combineer UFH KF en DHW EKF.
- [ ] Return `ThermalStateEstimate`.
- [ ] Garandeer dat `vdot_tap_hat` geclamped is.
- [ ] Voeg integratietest toe voor estimator output naar MPC.

---

# Fase 4 — Forecast-laag

## 4.1 Raw forecast datamodellen
- [ ] Implementeer `RawForecastBundle`.
- [ ] Velden:
  - `t_out`
  - `gti_window`
  - `q_int_base`
  - `t_mains`
  - `t_amb_tank`
  - `price_eur_per_kwh`
  - `vdot_tap_forecast`
  - comfortprofielen
- [ ] Valideer gelijke lengtes.
- [ ] Valideer geen ontbrekende waarden.
- [ ] Valideer fysieke grenzen.

## 4.2 Multi-resolution horizon
- [ ] Implementeer horizon builder:
  - 12h @ 15 min
  - 36h @ 60 min
- [ ] Maak `delta_t_hours[k]`.
- [ ] Zorg dat alle forecast arrays lengte `N=84` hebben.
- [ ] Voeg test toe:
  - `test_multiresolution_horizon_shape`

## 4.3 Forecast correction layer
- [ ] Implementeer `ForecastCorrectionLayer`.
- [ ] Corrigeer zonforecast met bias/gain.
- [ ] Corrigeer interne load met biasprofiel.
- [ ] Corrigeer tapload met EKF-output + historisch profiel.
- [ ] Implementeer horizon-afhankelijke decay:
  - sterke nowcastcorrectie 0–2h
  - uitdoving 2–12h
  - profielgebaseerd 12–48h
- [ ] Clamp:
  - `GTI_window >= 0`
  - `Q_solar >= 0`
  - `Vdot_tap >= 0`
- [ ] Voeg tests toe:
  - `test_forecast_correction_decay`
  - `test_solar_forecast_nonnegative`
  - `test_tap_forecast_nonnegative`

## 4.4 Residual tracker
- [ ] Implementeer `ModelResidualTracker`.
- [ ] Bereken UFH residual:
  - gemeten `T_r` minus predicted `T_r`
- [ ] Bereken DHW residuals:
  - `T_top`
  - `T_bot`
- [ ] Gebruik residuals alleen voor correctie, niet direct voor parameterdrift.
- [ ] Voeg diagnostics toe.

---

# Fase 5 — Warmtepomp performance learning

## 5.1 Performance samples
- [ ] Implementeer `HeatPumpPerformanceSample`.
- [ ] Log:
  - mode
  - `T_out`
  - `T_flow`
  - `T_return`
  - `T_top`
  - `T_bottom`
  - elektrisch vermogen
  - thermisch vermogen indien beschikbaar
  - compressorstatus
  - defrost/fault status
- [ ] Implementeer afleiding van `P_th_measured`:
  - directe warmtemeter
  - of hydraulische afleiding
  - of energiebalans-surrogaat

## 5.2 Feature extraction
- [ ] Implementeer `FeatureExtractor`.
- [ ] Bouw features:
  - `t_cond_proxy`
  - `t_evap_proxy`
  - `delta_t_lift`
  - `part_load_ratio`
  - mode-specifieke temperaturen
- [ ] Bouw features voor actuele samples.
- [ ] Bouw features voor forecast horizon.

## 5.3 Learning update gate
- [ ] Implementeer `LearningUpdateGate`.
- [ ] Accepteer alleen samples als:
  - mode niet `OFF`
  - compressor stabiel aan
  - geen defrost
  - geen fault
  - geen ontbrekende essentiële signalen
  - `P_el > 0`
  - sample lang genoeg stabiel
- [ ] Log reject-reasons.
- [ ] Voeg tests toe:
  - `test_invalid_sample_does_not_update_model`

## 5.4 Performance model
- [ ] Implementeer `OfflineLinearPerformanceModel`.
- [ ] Implementeer `BiasAdaptivePerformanceModel`.
- [ ] Houd aparte modellen voor:
  - `UFH` capacity
  - `DHW` capacity
  - `UFH` COP
  - `DHW` COP
- [ ] Implementeer clamps:
  - `p_th_max_min <= p_th_max <= p_th_max_physical`
  - `cop_min_physical < cop <= cop_max`
- [ ] Implementeer fallback bij:
  - te weinig data
  - OOD
  - numerieke fout
- [ ] Voeg tests toe:
  - `test_cop_clamp`
  - `test_power_envelope_clamp`
  - `test_learning_fallback_on_invalid_prediction`

## 5.5 Performance envelope provider
- [ ] Implementeer `PerformanceEnvelopeProvider`.
- [ ] Lever horizon arrays:
  - `p_th_max_ufh[k]`
  - `p_th_max_dhw[k]`
  - `cop_ufh[k]`
  - `cop_dhw[k]`
- [ ] Integreer met forecast bundle.
- [ ] Voeg test toe:
  - `test_performance_envelope_shapes`
  - `test_learned_envelope_reaches_mpc_bounds`

---

# Fase 6 — Supervisor en modeplanning

## 6.1 Basistypes
- [ ] Implementeer `HeatPumpMode`.
- [ ] Implementeer `HeatPumpModeContext`.
- [ ] Implementeer `ModeBlock`.
- [ ] Implementeer `ModePlanCandidate`.
- [ ] Implementeer `ActuatorAvailability`.

## 6.2 HeatPumpTopologySupervisor
- [ ] Implementeer exclusieve topologieregel:
  - nooit UFH en DHW tegelijk
- [ ] Implementeer minimum on-time.
- [ ] Implementeer minimum off-time.
- [ ] Implementeer minimum UFH dwell.
- [ ] Implementeer minimum DHW dwell.
- [ ] Implementeer switch-policy:
  - productiedefault: ramp vrijstellen op switch-stap
- [ ] Voeg tests toe:
  - `test_exclusive_heat_pump_topology_requires_supervisor_mode`
  - `test_min_on_time_enforced`
  - `test_min_off_time_enforced`
  - `test_dwell_enforced`
  - `test_exclusive_mode_switch_has_ramp_policy`

## 6.3 LegionellaSupervisor
- [ ] Implementeer `LegionellaRequest`.
- [ ] Detecteer of legionella-run nodig is.
- [ ] Plan DHW-window binnen horizon.
- [ ] Vereis beide nodes:
  - `T_top >= T_leg_target`
  - `T_bot >= T_leg_target`
- [ ] Voeg tests toe:
  - `test_legionella_surrogate_uses_both_nodes`

## 6.4 ModePlanGenerator
- [ ] Genereer beperkte kandidaatset.
- [ ] Templates:
  - stay current
  - switch to off
  - switch to UFH
  - switch to DHW
  - UFH then OFF
  - DHW then OFF
  - OFF then UFH
  - OFF then DHW
  - UFH then DHW
  - DHW then UFH
  - urgent room recovery
  - urgent DHW recovery
  - legionella plan
- [ ] Prune kandidaten die regels schenden.
- [ ] Voeg max switch count toe.
- [ ] Voeg test toe:
  - `test_candidate_generation_prunes_invalid_plans`

## 6.5 Availability masks
- [ ] Bouw `p_ufh_min/max` en `p_dhw_min/max`.
- [ ] OFF:
  - beide nul
- [ ] UFH:
  - DHW nul
  - UFH begrensd door mode + performance envelope
- [ ] DHW:
  - UFH nul
  - DHW begrensd door mode + performance envelope
- [ ] Voeg test toe:
  - `test_availability_masks_match_mode_plan`

---

# Fase 7 — MPC-laag

## 7.1 COP precalculator
- [ ] Implementeer `CopPrecalculator`.
- [ ] Ondersteun vaste COP-map.
- [ ] Ondersteun learned COP-envelope.
- [ ] Valideer:
  - `T_cond > T_evap`
  - `T_evap > 0`
  - `cop_min_physical < COP <= cop_max`
- [ ] Voeg test toe:
  - `test_cop_validation`

## 7.2 MpcProblemBuilder
- [ ] Implementeer gecombineerde state:
  - `[T_r, T_b, T_top, T_bot]`
- [ ] Implementeer inputs:
  - `P_ufh`
  - `P_dhw`
- [ ] Implementeer slackvariabelen:
  - `eps_ufh`
  - `eps_dhw`
- [ ] Bouw dynamica per horizonstap.
- [ ] Gebruik `delta_t_hours[k]`.
- [ ] Voeg availability constraints toe.
- [ ] Voeg comfort constraints toe.
- [ ] Voeg safety bounds toe.
- [ ] Voeg DHW terminal policy toe.
- [ ] Voeg explicit previous input parameters toe:
  - `P_ufh_prev`
  - `P_dhw_prev`
- [ ] Voeg ramp-rate constraints toe, behalve switch-stappen volgens policy.
- [ ] Voeg OSQP solve interface toe.
- [ ] Voeg tests toe:
  - `test_mpc_feasibility_nominal`
  - `test_temperature_safety_bounds_enforced`
  - `test_mpc_requires_explicit_previous_input_parameters`
  - `test_dhw_terminal_strategy_present_or_explicitly_disabled`

## 7.3 Cost function
- [ ] Implementeer stage cost:
  - room comfort
  - UFH electricity
  - DHW electricity
  - UFH power penalty
  - slack penalties
- [ ] Vermenigvuldig stage-termen met `delta_t_hours[k]`.
- [ ] Implementeer room terminal penalty.
- [ ] Implementeer DHW terminal penalty/bound.
- [ ] Voeg plan-cost buiten QP toe:
  - `w_switch_count * switch_count`
  - `w_start_count * start_count`
- [ ] Voeg test toe:
  - `test_cost_uses_delta_t`

## 7.4 ModePlanEvaluator
- [ ] Implementeer loop over kandidaten.
- [ ] Bouw availability masks.
- [ ] Bouw MPC-probleem.
- [ ] Los OSQP op.
- [ ] Verwerp infeasible kandidaten.
- [ ] Tel plan-cost op bij QP-cost.
- [ ] Kies goedkoopste haalbare kandidaat.
- [ ] Voeg fallback toe als geen kandidaat haalbaar is:
  - veilige comfort-preserving rule
  - of hard exception afhankelijk van config
- [ ] Voeg tests toe:
  - `test_infeasible_candidate_rejected`
  - `test_best_feasible_candidate_selected`

---

# Fase 8 — Control loop

## 8.1 Hoofdcontroller
- [ ] Implementeer `HomeOptimizerController`.
- [ ] Control-step flow:
  1. lees metingen
  2. update estimator
  3. update learning met vorige sample
  4. bouw raw forecast
  5. corrigeer forecast
  6. bouw performance envelope
  7. genereer legionella request
  8. genereer mode candidates
  9. evalueer kandidaten met MPC
  10. kies beste
  11. voer eerste actie uit
  12. update context
  13. log diagnostics
- [ ] Voeg integratietest toe:
  - `test_control_step_returns_safe_action`

## 8.2 Actuator decision
- [ ] Implementeer `ControlDecision`.
- [ ] Velden:
  - selected mode
  - `P_ufh_set`
  - `P_dhw_set`
  - first-step cost
  - candidate description
  - diagnostics
- [ ] Valideer:
  - geen negatieve vermogens
  - geen simultane UFH/DHW bij exclusieve topologie
  - geen negatieve tapstroom naar MPC

---

# Fase 9 — Diagnostics, logging en monitoring

## 9.1 Logging per control-step
- [ ] Log state estimate.
- [ ] Log forecast corrections.
- [ ] Log performance envelope.
- [ ] Log candidate list.
- [ ] Log rejected candidates met reason.
- [ ] Log OSQP status per candidate.
- [ ] Log selected mode plan.
- [ ] Log switch/start count.
- [ ] Log slack usage.
- [ ] Log comfort violations.
- [ ] Log legionella status.
- [ ] Log learning diagnostics.

## 9.2 Diagnostics objects
- [ ] Implementeer `EstimatorDiagnostics`.
- [ ] Implementeer `ForecastDiagnostics`.
- [ ] Implementeer `LearningDiagnostics`.
- [ ] Implementeer `SupervisorDiagnostics`.
- [ ] Implementeer `MpcDiagnostics`.

## 9.3 Fail-fast exceptions
- [ ] Maak exception types:
  - `InvalidConfigurationError`
  - `PhysicalValidationError`
  - `ForecastValidationError`
  - `ObservabilityError`
  - `CovarianceValidationError`
  - `NoFeasibleModePlanError`
  - `SolverFailureError`

---

# Fase 10 — Testscenario’s

## 10.1 Unit tests
- [ ] Configvalidatie.
- [ ] Modelmatrices.
- [ ] Discretisatie.
- [ ] Energiebalans.
- [ ] Observability.
- [ ] Covariance PSD.
- [ ] Forecast correction.
- [ ] Learning gate.
- [ ] Supervisor transition rules.
- [ ] Availability masks.
- [ ] MPC constraints.

## 10.2 End-to-end scenario’s
- [ ] Winterdag zonder tapvraag.
- [ ] Ochtend-DHW draw.
- [ ] Goedkope stroom in nacht → preheat.
- [ ] Hoge stroomprijs overdag → avoid unless needed.
- [ ] Ruimte dreigt af te koelen → UFH recovery.
- [ ] Tank dreigt onder comfort → DHW recovery.
- [ ] Legionella-run.
- [ ] Defrost/invalid learning sample.
- [ ] Forecast solar overestimated.
- [ ] Tapload forecast underestimated.
- [ ] Exclusieve mode-switch met ramp-policy.

## 10.3 Regressietests
- [ ] Geen simultane UFH/DHW bij exclusieve topologie.
- [ ] DHW nooit als één state voorspeld.
- [ ] `Vdot_tap_hat` altijd nonnegative naar MPC.
- [ ] Learned COP nooit buiten grenzen.
- [ ] Geen brute-force combinatoriek.
- [ ] No magic numbers.

---

# Fase 11 — Implementatievolgorde voor Codex

## Milestone 1 — Physics core
- [ ] Configs.
- [ ] UFH-model.
- [ ] DHW-model.
- [ ] Discretizer.
- [ ] Energiebalans-tests.

## Milestone 2 — Estimation
- [ ] KF.
- [ ] EKF.
- [ ] StateEstimatorFacade.
- [ ] PSD/observability tests.

## Milestone 3 — Basic MPC
- [ ] ForecastBundle.
- [ ] Fixed-mode MPC.
- [ ] Safety/comfort constraints.
- [ ] DHW terminal policy.
- [ ] Nominal feasibility.

## Milestone 4 — Supervisor
- [ ] Modes.
- [ ] Context.
- [ ] TopologySupervisor.
- [ ] Minimum on/off.
- [ ] Dwell.
- [ ] Availability masks.

## Milestone 5 — Hybrid planning
- [ ] ModePlanGenerator.
- [ ] ModePlanEvaluator.
- [ ] Candidate ranking.
- [ ] OSQP per candidate.
- [ ] Best-plan selection.

## Milestone 6 — Forecast correction
- [ ] RawForecastBundle.
- [ ] ForecastCorrectionLayer.
- [ ] ResidualTracker.
- [ ] Solar/load/tap correction.

## Milestone 7 — Learning layer
- [ ] PerformanceSample.
- [ ] FeatureExtractor.
- [ ] LearningUpdateGate.
- [ ] OfflineLinearPerformanceModel.
- [ ] BiasAdaptivePerformanceModel.
- [ ] PerformanceEnvelopeProvider.

## Milestone 8 — Production hardening
- [ ] Logging.
- [ ] Diagnostics.
- [ ] Fallbacks.
- [ ] End-to-end tests.
- [ ] Performance profiling.
- [ ] Warm-start OSQP.