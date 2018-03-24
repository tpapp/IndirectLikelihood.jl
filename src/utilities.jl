"""
    $SIGNATURES

Informative error message for missing method.
"""
macro no_method_info(ex)
    @capture(ex, f_(args__)) || error("Expected a function name with arguments")
    msg = "You need to define `$(string(f))` with this model type."
    esc_f = esc(f)
    esc_args = map(esc, args)
    quote
        function $esc_f($(esc_args...))
            info($msg)
            throw(MethodError($esc_f,
                              Tuple{$(map(e -> :(typeof($e)), esc_args)...)}))
        end
    end
end
