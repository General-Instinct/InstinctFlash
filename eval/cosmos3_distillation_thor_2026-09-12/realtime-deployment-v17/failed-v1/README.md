V1 preparation accidentally declared num_train_timesteps=2000 when adapting
steps=1 to steps=2. Native weights/config remained unchanged; CPU packaging
completed, but the declaration is invalid for the qualified 1000/500 clock.
Caught during source diff review before any transfer or model execution.
The two v17-cfg1-seed* packages without the -v2 suffix are excluded. Preserve
these outputs as failed preparation evidence. V2 keeps num_train_timesteps=1000
and tests the actual generated declaration as well as export gate admission.
