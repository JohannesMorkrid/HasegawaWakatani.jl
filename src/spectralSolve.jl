# Assuming for now that dt is fixed
# If custom outputter is not provided, then resort to default
# First step is stored during initilization of output
function spectral_solve(prob::SOP, scheme::SA=MSS3(),
                        output::O=Output(prob; store_hdf=false); debug=false,
                        resume::Bool=false) where {SOP<:SpectralODEProblem,
                                                   SA<:AbstractODEAlgorithm,O<:Output}
    # Initialize cache and tracking
    cache, t, step = initialize_solve(prob, scheme, output, resume)

    # Time step
    dt = prob.dt

    # Calculate number of steps #TODO ceil or floor?
    total_steps = floor(Int, (last(prob.tspan) - first(prob.tspan)) / dt)

    # Enable CTRL+C from terminal outside of interactive mode
    Base.exit_on_sigint(false)

    # Asyncronous background process that backups the file and checkpoint
    backup_timer = Timer(_ -> backup(output, cache, step, t), 0;
                         interval=output.backup_interval)

    try
        # This method assumes step number does not overflow!
        while step < total_steps
            perform_step!(cache, prob, t)

            # Increment step and time
            step += 1
            t = first(prob.tspan) + step * dt

            handle_output!(output, step, cache.u, prob, t)
        end
    catch error
        # Interupt the error, so that the code does not halt when not in debug mode
        debug ? rethrow(error) : showerror(stdout, error)
    end

    # TODO catch edge case

    # Stop the backuping
    close(backup_timer)

    # Do one more backup for safety measure 
    backup(output, cache, step, t)

    # Returns output struct
    return output
end

function initialize_solve(prob::SOP, scheme::SA, output::O,
                          resume::Bool) where {
                                               SOP<:SpectralODEProblem,
                                               SA<:AbstractODEAlgorithm,O<:Output}
    if resume && output.store_hdf && haskey(output.simulation, "checkpoint")
        cache = restore_checkpoint(output.simulation, prob, scheme)
        t = read(output.simulation, "checkpoint/time")
        step = read(output.simulation, "checkpoint/step")
    else
        cache = get_cache(prob, scheme)
        t = first(prob.tspan)
        step = 0
    end
    return cache, t, step
end

function backup(output, cache, step, t)
    output.store_hdf ? nothing : return nothing
    # Store the cache to be able to resume simulations
    save_checkpoint!(output, cache, step, t)
    # Write buffer to file
    flush(output.simulation.file)
end