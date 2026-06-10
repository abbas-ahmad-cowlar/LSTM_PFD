# Backlog — Cut But Plausible Future Directions

> One line per idea. Anything here was deliberately cut during convergence (see
> `CONVERGENCE_PLAN.md`); code is recoverable from tag `pre-convergence-2026-06`.
> Nothing on this list may be started before Convergence Gate 5 (see plan Part IV rule 4).

- **Time-frequency input (TFR/2D)**: spectrogram datasets + ResNet2D/EfficientNet2D models — a future "does 2D input help?" study.
- **Contrastive pretraining**: SimCLR-style SignalEncoder pretraining vs supervised — a separate self-supervision paper.
- **Knowledge distillation for edge**: distill best PINN into a tiny CNN; pairs with the deployment appendix.
- **KnowledgeGraphPINN**: speculative physics-graph architecture; revisit only with a concrete formulation.
- **Sim-to-real transfer study**: validate on real journal-bearing data (industrial partner or test rig) — the natural follow-up paper.
- **Architecture breadth benchmark**: EfficientNet-1D/WideResNet/ViT-1D/TSMixer/CNN-TCN rows — leaderboard-style extension if ever useful.
- **Stacking/boosting/MoE ensembles + fusion models**: ensemble-strategy comparison study.
- **Docs site (mkdocs)**: rebuild after the paper, on top of the converged docs/.
- **K8s/Helm deployment**: only if a real multi-user deployment need appears.
- **SaaS productionization** (multi-tenancy, billing, GDPR): out of scope for a research project; revisit only as a separate product decision.
- **NAS / HPO dashboards**: automated search infrastructure; needs the platform to be stable first.
- **Dashboard advanced features** (webhooks, 2FA, API keys, notifications): blocked on Phase D login decision.
