
Engine Architecture
------------------------

The ``Engine`` in **RoboOrchardLab** orchestrates the training and evaluation processes through a well-defined pipeline.
This pipeline is highly extensible via a system of hooks that can be triggered at various stages.
Understanding this pipeline and its hook points is key to customizing and extending the framework's behavior.

The following flowchart illustrates the primary execution flow, with a focus on when different hook channels are activated:

.. image:: ../_static/overview/engine_architecture.png
   :alt: Engine Architecture Diagram
   :align: center
   :width: 100%


The whole training process with hooks is as follows:

.. code-block:: text

    with on_loop:
        with on_epoch:
            for batch in dataloader:
                with on_step:
                    with on_batch:
                        batch_processor(...)
                        ...

                    update step id
        update epoch id


Hook Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline is structured as a series of nested loops and contexts, each providing specific hook channels for user intervention:

1.  **Overall Run Context (``on_loop``)**:

    * ``hook: on_loop_begin``: Called once at the very beginning of the entire run (e.g., `engine()` call). Ideal for global setup, like initializing external loggers or loading comprehensive resources.
    * *(Main execution of epochs occurs here)*
    * ``hook: on_loop_end``: Called once at the very end of the entire run. Suitable for final cleanup, saving summary reports, or closing external services.

2.  **Epoch Context (``on_epoch``)**: This loop iterates for the configured number of epochs.

    * ``hook: on_epoch_begin``: Called at the start of each epoch. Useful for epoch-specific setup, resetting epoch-level metrics, or adjusting dynamic parameters.
    * *(Main execution of steps/batches within the epoch occurs here)*
    * ``hook: on_epoch_end``: Called at the end of each epoch. This is a common place to:
        * Run an evaluation phase (see "Evaluation Pipeline" below).
        * Perform learning rate scheduler steps (for epoch-based schedulers).
        * Save epoch-level checkpoints.
        * Aggregate and log epoch-wide metrics.

3.  **Step Context (Optimizer Step - ``on_step``)**: This loop typically corresponds to one optimizer update (an "optimizer step"). When using gradient accumulation, one "step" can encompass multiple "batch" (micro-batch) forward/backward passes.

    * ``hook: on_step_begin``: Called at the beginning of an optimizer step, before processing any micro-batches for this step.
    * *(Processing of one or more micro-batches occurs here)*
    * ``hook: on_step_end``: Called after all micro-batches for the current optimizer step have been processed and typically after the optimizer has updated the model weights (e.g., after ``optimizer.step()`` and ``optimizer.zero_grad()``). Good for step-level logging, and some LR scheduler updates.

4.  **Batch Context (Micro-batch Processing - ``on_batch``)**: This is the innermost loop if gradient accumulation is used, or it might be equivalent to the "step" loop if accumulation is not active (i.e., 1 batch = 1 optimizer step). It processes a single batch of data through the model.

    * ``hook: on_batch_begin``: Called before the forward and backward pass of a single batch.
    * *(Model forward and backward passes occur here)*
    * ``hook: on_batch_end``: Called after the forward and backward pass of a single batch.

5.  **Model Execution Contexts**: These hooks provide fine-grained control around the model's forward and backward passes for each batch.

    * ``hook: on_model_forward_begin``: Called just before ``batch_processor: forward``.
    * *(``batch_processor: forward`` executes model's forward pass and loss computation)*
    * ``hook: on_model_forward_end``: Called just after ``batch_processor: forward``.
    * ``hook: on_model_backward_begin``: Called just before ``batch_processor: backward`` (which internally calls ``accelerator.backward(loss)``).
    * *(``batch_processor: backward`` executes the backward pass)*
    * ``hook: on_model_backward_end``: Called just after ``batch_processor: backward``.

6. **Extensibility**: This structured pipeline with its numerous hook channels allows for significant customization. Users can inject their own logic for metric calculation, logging, checkpointing, learning rate scheduling, early stopping, custom data manipulations, and much more, without altering the core ``Engine`` code. This makes RoboOrchardLab highly adaptable to diverse research needs.


Batch Processor Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **``batch_processor: forward``**: This is where the model processes the input batch, and typically, the loss is computed.
* **``batch_processor: backward``**: This stage performs the backpropagation using the computed loss. The actual ``optimizer.step()`` and ``optimizer.zero_grad()`` are managed by the ``Engine`` in conjunction with the ``on_step_*`` hooks and ``Accelerator`` (especially when using gradient accumulation via ``accelerator.accumulate()``).
