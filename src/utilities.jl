"""
    $SIGNATURES

Informative error message for missing method.
"""
@inline function no_model_method(f, args...)
    info("You need to define `$(string(f))` with this model type.")
    throw(MethodError(f, args))
end
