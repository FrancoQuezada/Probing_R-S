# Sensitivity of eq. (2), Notes on Softmax, to beta, for the unchanged ex-ante
# routing objective J(z)=max_i E_z[Y_i]-alpha'z. No controls/variance reduction
# are added to the search estimator. All inherited definitions remain untouched.
module RoutingSensitivityCore
include(joinpath(@__DIR__,"003probing_four_searches_softmax.jl"))
end
using .RoutingSensitivityCore
using Random, LinearAlgebra, Statistics, Printf, Serialization, TOML, SHA, Dates
using Plots
const C = RoutingSensitivityCore
const DEFAULT_BETAS = [0.025,0.05,0.10,0.20,0.50,1.00]
const DEFAULT_OUTPUT = joinpath(@__DIR__,"resultados/004probing_softmax_sensitivity")

function configuration(;betas=DEFAULT_BETAS,output=DEFAULT_OUTPUT,smoke=false,reuse=true,
                       references=joinpath(@__DIR__,"resultados/003probing_softmax_20261021/completo"),
                       final_seed=2026102101)
    bs=sort(unique(Float64.(betas)))
    !isempty(bs) && all(b->isfinite(b)&&b>0,bs) || error("Betas must be finite and positive")
    return (betas=bs,output=abspath(output),smoke=smoke,reuse=reuse,references=abspath(references),
            npaths=smoke ? 30 : C.N_PATHS,njobs=smoke ? 12 : C.JOBS_PER_PATH,
            iterations=smoke ? 4 : C.MAX_ITERS,trace_paths=smoke ? 40 : C.TRACE_EVAL_PATHS,
            final_paths=smoke ? 100 : C.ORACLE_PATHS_FINAL,trace_seed=C.TRACE_EVAL_SEED,
            final_seed=Int(final_seed),search_seed=C.SEED_FRESH,epsilon=1e-10,
            grid_n=smoke ? 3 : C.ORACLE_GRID_N,surface_paths=smoke ? 40 : C.ORACLE_PATHS_PER_GRID_POINT,
            surface_seed=C.ORACLE_SURFACE_SEED)
end

function parse_arguments(args)
    betas=copy(DEFAULT_BETAS); output=DEFAULT_OUTPUT; smoke=false; reuse=true
    references=joinpath(@__DIR__,"resultados/003probing_softmax_20261021/completo")
    final_seed=2026102101; i=1; explicit_output=false
    while i<=length(args)
        a=args[i]
        if a=="--smoke"; smoke=true
        elseif a=="--no-reuse"; reuse=false
        elseif a in ("--betas","--output","--references","--final-seed")
            i+=1; i<=length(args) || error("Missing value for $a")
            if a=="--betas"; betas=parse.(Float64,split(args[i],','))
            elseif a=="--output"; output=args[i]; explicit_output=true
            elseif a=="--references"; references=args[i]
            else; final_seed=parse(Int,args[i]); end
        elseif a=="--help"
            println("Usage: julia 004probing_softmax_sensitivity.jl [--betas 0.025,0.05,0.1,0.2,0.5,1] [--output DIR] [--smoke] [--no-reuse] [--references DIR] [--final-seed INT]")
            return nothing
        else; error("Unknown option $a"); end
        i+=1
    end
    smoke && !explicit_output && (output=joinpath(DEFAULT_OUTPUT,"prueba"))
    return configuration(;betas,output,smoke,reuse,references,final_seed)
end

# Exactly the original primitive RNG sequence, with a controllable TEST horizon.
function primitive_bank(cfg,n,seed)
    rng=MersenneTwister(seed)
    return [C.generate_crn_primitive_path(rng;njobs=cfg.njobs) for _ in 1:n]
end

"""One bank, one Y and one score matrix for BOTH gradients at the SAME z."""
function local_policy_data(z,bank,policies)
    Y=C.precompute_rewards(policies,bank)
    n=length(bank); scores=Matrix{Float64}(undef,2,n)
    for l in eachindex(bank)
        ll,sc=C.loglr_and_score(bank[l],z,z)
        ll==0 || error("Local LR is not exactly one")
        scores[:,l].=sc
    end
    values=vec(mean(Y;dims=2))
    G=(Y*transpose(scores))/n
    best=argmax(values)
    # Match original Fresh's summation, including its numerical reduction order.
    hard=[mean(@view(Y[best,:]).*@view(scores[j,:]))-C.ALPHA[j] for j in 1:2]
    all(isfinite,G) && all(isfinite,hard) || error("Nonfinite gradient")
    return (values=values,G=G,hard=hard,best=best,objective=values[best]-dot(C.ALPHA,z))
end

function safe_cosine(a,b,eps)
    na=norm(a); nb=norm(b)
    status=na<=eps && nb<=eps ? "both_near_zero" : na<=eps ? "soft_near_zero" : nb<=eps ? "hard_near_zero" : "defined"
    return status=="defined" ? clamp(dot(a,b)/(na*nb),-1.0,1.0) : missing,status
end

function smoothing_diagnostics(data,z,k,beta,epsilon)
    if ismissing(beta)
        w=zeros(length(data.values));w[data.best]=1
        g=copy(data.hard)
    else
        comb=C.combine_softmax_gradients(data.values,data.G;beta=beta)
        w=comb.weights;g=comb.gradient
    end
    @assert all(isfinite,w) && all(>=(0),w) && isapprox(sum(w),1;atol=1e-12)
    hard=data.hard
    znew,eta,gc=C.projected_step(z,g,k)
    hardnew,_,hc=C.projected_step(z,hard,k)
    cosine,status=safe_cosine(g,hard,epsilon)
    clipped_cosine,clipped_status=safe_cosine(gc,hc,epsilon)
    rawgap=norm(g-hard);clipgap=norm(gc-hc);projectedgap=norm(znew-hardnew)
    spread=sqrt(sum(w[i]*sum(abs2,view(data.G,i,:).-view(data.G,data.best,:)) for i in eachindex(w)))
    # Weighted distance to the best policy gradient is descriptive only; no centering
    # or control variate is applied to the estimator used in projected_step.
    return (gradient=g,znew=znew,
        diagnostics=(iteration=k,z3=z[1],z4=z[2],J_internal=data.objective,policy=data.best,
        g3=g[1],g4=g[2],hard_g3=hard[1],hard_g4=hard[2],g_norm=norm(g),hard_g_norm=norm(hard),
        clipped_g3=gc[1],clipped_g4=gc[2],hard_clipped_g3=hc[1],hard_clipped_g4=hc[2],
        K_eff=inv(sum(abs2,w)),max_weight=maximum(w),weight_over_1pct=count(>(0.01),w),
        entropy=-sum(x>0 ? x*log(x) : 0.0 for x in w),weight_sum=sum(w),
        relative_difference=rawgap/max(norm(hard),epsilon),gradient_difference=rawgap,
        cosine=cosine,cosine_status=status,clipped_cosine=clipped_cosine,clipped_cosine_status=clipped_status,
        clipped_gradient_difference=clipgap,relative_clipped_difference=clipgap/max(norm(hc),epsilon),
        soft_clipping_change=norm(g-gc),hard_clipping_change=norm(hard-hc),
        clipped_components=count(x->abs(x)>C.GRAD_CLIP,g),
        soft_projection_change=norm(znew-(z+eta*gc)),hard_projection_change=norm(hardnew-(z+eta*hc)),
        step_norm=norm(znew-z),hard_step_norm=norm(hardnew-z),projected_update_difference=projectedgap,
        raw_update_difference=eta*rawgap,clipped_update_difference=eta*clipgap,
        clipping_erases_difference=rawgap>epsilon && clipgap<=epsilon,
        projection_erases_difference=clipgap>epsilon && projectedgap<=epsilon,
        weighted_policy_gradient_spread=spread/max(norm(hard),epsilon),
        weighted_reward_gap=dot(w,maximum(data.values).-data.values),eta=eta))
end

function load_reference(cfg)
    cfg.reuse && !cfg.smoke || return nothing
    mp=joinpath(cfg.references,"probing_four_metadata.toml")
    rp=joinpath(cfg.references,"probing_four_results.jls")
    if !(isfile(mp)&&isfile(rp));println("No compatible reference files; Fresh and all requested betas will run.");return nothing;end
    m=TOML.parsefile(mp)
    required=Dict("N_PATHS"=>cfg.npaths,"JOBS_PER_PATH"=>cfg.njobs,"MAX_ITERS"=>cfg.iterations,
      "TRACE_EVAL_PATHS"=>cfg.trace_paths,"TRACE_EVAL_SEED"=>cfg.trace_seed,
      "ORACLE_GRID_N"=>cfg.grid_n,"ORACLE_PATHS_PER_GRID_POINT"=>cfg.surface_paths,
      "ORACLE_SURFACE_SEED"=>cfg.surface_seed,"BETA"=>C.BETA,"ALPHA"=>C.ALPHA,"Z0"=>C.Z0,
      "BASE_STEP"=>C.BASE_STEP,"GRAD_CLIP"=>C.GRAD_CLIP,"SEED_FRESH"=>cfg.search_seed,
      "julia_version"=>string(VERSION),"blas_threads"=>BLAS.get_num_threads(),
      "source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"003probing_four_searches_softmax.jl")))))
    for (key,value) in required
        get(m,key,nothing)==value || error("Reference mismatch for $key; use --no-reuse or a compatible reference directory")
    end
    ref=deserialize(rp)
    @assert length(ref.searches[2].zhist)==cfg.iterations+1
    println("Reusing exact Fresh and beta=0.5 searches/traces; Fixed/Green references and oracle surface.")
    return ref
end

method_name(beta)=ismissing(beta) ? "Fresh" : "Softmax beta=$(beta)"
function run_searches(cfg,policies;reference=nothing,order=nothing)
    betas=Union{Missing,Float64}[missing;cfg.betas]
    states=map(betas) do beta
        old=reference===nothing ? nothing : ismissing(beta) ? reference.searches[2] : beta==0.5 ? reference.searches[4] : nothing
        hist=old===nothing ? [copy(C.Z0)] : deepcopy(old.zhist)
        (method=method_name(beta),beta=beta,zhist=hist,diagnostics=NamedTuple[],reused=old!==nothing,old=old)
    end
    indices=order===nothing ? collect(eachindex(states)) : order
    sort(indices)==collect(eachindex(states)) || error("Invalid execution order")
    anchors=NamedTuple[]; bank_evaluations=0
    for k in 1:cfg.iterations
        seed=cfg.search_seed+10000*k
        primitives=primitive_bank(cfg,cfg.npaths,seed) # ONE underlying noise bank per iteration
        payloads=Dict{Tuple{Float64,Float64},Any}()
        function payload(z)
            get!(payloads,Tuple(z)) do
                bank=C.materialize_crn_bank(primitives,z)
                local_policy_data(z,bank,policies)
            end
        end
        # Fixed-Z comparison across betas disambiguates beta from differing trajectories.
        zfresh=states[1].zhist[k]; freshdata=payload(zfresh)
        previous_keff=Inf
        for beta in cfg.betas
            d=smoothing_diagnostics(freshdata,zfresh,k,beta,cfg.epsilon).diagnostics
            @assert d.K_eff<=previous_keff+1e-10
            previous_keff=d.K_eff
            push!(anchors,merge((beta=beta,seed=seed),d))
        end
        for i in indices
            state=states[i];z=state.zhist[k];data=payload(z)
            est=smoothing_diagnostics(data,z,k,state.beta,cfg.epsilon)
            if state.reused
                @assert isapprox(data.objective,state.old.Jhist[k];atol=1e-10,rtol=1e-12)
                @assert isapprox(est.znew,state.zhist[k+1];atol=1e-12,rtol=1e-12)
                if !ismissing(state.beta)
                    @assert isapprox(est.gradient,state.old.diagnostics[k].gradient;atol=1e-11,rtol=1e-12)
                end
            else
                push!(state.zhist,est.znew)
            end
            @assert all(x->0<=x<=1,est.znew)
            push!(state.diagnostics,merge((seed=seed,),est.diagnostics))
        end
        bank_evaluations+=length(payloads)
        if k==1 || k%10==0
            @printf("Search/diagnostics iteration %d/%d, unique point banks this iteration=%d\n",k,cfg.iterations,length(payloads));flush(stdout)
        end
    end
    runs=[(method=s.method,beta=s.beta,zhist=s.zhist,diagnostics=s.diagnostics,reused=s.reused) for s in states]
    return runs,anchors,bank_evaluations
end

function evaluate_traces(cfg,runs,policies,reference)
    bank=primitive_bank(cfg,cfg.trace_paths,cfg.trace_seed) # ONE common CRN object
    cache=Dict{Tuple{Float64,Float64},Float64}()
    if reference!==nothing
        for i in (1,2,3,4), k in eachindex(reference.searches[i].zhist)
            cache[Tuple(reference.searches[i].zhist[k])]=reference.traces[i][k]
        end
    end
    new_points=0
    traces=map(runs) do run
        unknown=Vector{Vector{Float64}}();seen=Set{Tuple{Float64,Float64}}()
        for z in run.zhist
            key=Tuple(z)
            if !haskey(cache,key) && !(key in seen);push!(unknown,z);push!(seen,key);end
        end
        for start in 1:10:length(unknown)
            points=unknown[start:min(start+9,length(unknown))]
            values=C.evaluate_trace_on_common_crn(points,policies,bank)
            for (z,value) in zip(points,values);cache[Tuple(z)]=value;end
            new_points+=length(points)
            @printf("CRN trace %s: %d/%d new points\n",run.method,min(start+9,length(unknown)),length(unknown));flush(stdout)
        end
        [cache[Tuple(z)] for z in run.zhist]
    end
    @assert all(t->t[1]==traces[1][1],traces)
    return traces,new_points
end

function reference_surface(cfg,policies,reference)
    reference!==nothing && return reference.grid,reference.surface,true
    println("No compatible surface cache; constructing a direct CRN surface for this configuration.")
    bank=primitive_bank(cfg,cfg.surface_paths,cfg.surface_seed)
    grid=collect(range(0,1;length=cfg.grid_n));Z=zeros(cfg.grid_n,cfg.grid_n)
    for (ix,z3) in enumerate(grid),(iy,z4) in enumerate(grid)
        z=[z3,z4];Y=C.precompute_rewards(policies,C.materialize_crn_bank(bank,z))
        Z[iy,ix]=maximum(vec(mean(Y;dims=2)))-dot(C.ALPHA,z)
    end
    return grid,Z,false
end

function evaluate_endpoints(cfg,runs,policies)
    bank=primitive_bank(cfg,cfg.final_paths,cfg.final_seed) # independent of traces and optimization
    values=Vector{Vector{Float64}}();rows=NamedTuple[]
    for run in runs
        z=run.zhist[end];Y=C.precompute_rewards(policies,C.materialize_crn_bank(bank,z))
        means=vec(mean(Y;dims=2));best=argmax(means)
        selected=collect(view(Y,best,:));push!(values,selected)
        row=(method=run.method,beta=run.beta,z3=z[1],z4=z[2],objective=means[best]-dot(C.ALPHA,z),
             gross=means[best],policy=best,SE=std(selected)/sqrt(cfg.final_paths))
        push!(rows,row)
        @printf("Final CRN %s: z=(%.8f,%.8f), J=%.8f, policy=%d, SE=%.6f\n",run.method,z...,row.objective,best,row.SE);flush(stdout)
    end
    pairs=NamedTuple[]
    for i in 1:length(runs)-1,j in i+1:length(runs)
        diffs=values[i]-values[j]
        delta=rows[i].objective-rows[j].objective
        se=std(diffs)/sqrt(cfg.final_paths)
        push!(pairs,(method_a=runs[i].method,method_b=runs[j].method,difference=delta,
                     paired_SE=se,CI95_low=delta-1.96se,CI95_high=delta+1.96se))
    end
    return rows,pairs,hcat(values...)
end

finite_mean(xs)=isempty(xs) ? missing : mean(xs)
function aggregate_results(cfg,runs,traces,oracle)
    map(eachindex(runs)) do i
        r=runs[i];d=r.diagnostics;t=traces[i];best=argmax(t)
        tail=t[max(2,length(t)-19):end] # last 20 UPDATED decisions, iterations 81..100
        cosines=Float64[x.cosine for x in d if !ismissing(x.cosine)]
        raw=[x.gradient_difference for x in d];post=[x.projected_update_difference for x in d]
        (method=r.method,beta=r.beta,z3=r.zhist[end][1],z4=r.zhist[end][2],
         J_initial=t[1],J_final=t[end],J_best=maximum(t),best_iteration=best-1,
         J_last20_mean=mean(tail),J_last20_sd=std(tail),
         total_variation=sum(norm(r.zhist[k+1]-r.zhist[k]) for k in 1:cfg.iterations),
         K_eff_mean=mean(x.K_eff for x in d),K_eff_median=median([x.K_eff for x in d]),
         relative_difference_mean=mean(x.relative_difference for x in d),
         relative_difference_median=median([x.relative_difference for x in d]),
         cosine_mean=finite_mean(cosines),undefined_cosine_count=length(d)-length(cosines),
         max_weight_mean=mean(x.max_weight for x in d),weight_over_1pct_mean=mean(x.weight_over_1pct for x in d),
         entropy_mean=mean(x.entropy for x in d),
         clipped_components=sum(x.clipped_components for x in d),
         projection_active_steps=count(x->x.soft_projection_change>cfg.epsilon,d),
         step_mean=mean(x.step_norm for x in d),projected_update_difference_mean=mean(post),
         clipping_erases_steps=count(x->x.clipping_erases_difference,d),
         projection_erases_steps=count(x->x.projection_erases_difference,d),
         policy_gradient_spread_mean=mean(x.weighted_policy_gradient_spread for x in d),
         oracle_objective=oracle[i].objective,oracle_SE=oracle[i].SE,oracle_policy=oracle[i].policy,
         reused=r.reused)
    end
end

csv_value(x::Missing)=""
csv_value(x::AbstractFloat)=@sprintf("%.17g",x)
csv_value(x)=occursin(r"[,\"\n]",string(x)) ? "\""*replace(string(x),"\""=>"\"\"")*"\"" : string(x)
function write_csv(path,rows)
    isempty(rows) && return
    keys=propertynames(first(rows))
    open(path,"w") do io
        println(io,join(string.(keys),','))
        for row in rows;println(io,join((csv_value(getproperty(row,k)) for k in keys),','));end
    end
end

const COLORS=[:black,:royalblue,:deepskyblue3,:forestgreen,:darkorange,:purple,:firebrick]
const STYLES=[:solid,:solid,:dash,:dot,:dashdot,:dash,:solid]
function line_figure(runs,series;xlabel="Iteration",ylabel="",title="",omit_fresh=false)
    p=plot(;xlabel,ylabel,title,size=(1150,750),left_margin=12Plots.mm,
           bottom_margin=10Plots.mm,top_margin=8Plots.mm,right_margin=7Plots.mm,
           titlefontsize=12,legendfontsize=9,legend=:bottomright)
    for (i,(run,values)) in enumerate(zip(runs,series))
        omit_fresh && i==1 && continue
        firstx=length(values)==length(run.zhist) ? 0 : 1
        plot!(p,firstx:firstx+length(values)-1,values;label=run.method,color=COLORS[mod1(i,length(COLORS))],
              linestyle=STYLES[mod1(i,length(STYLES))],lw=i==1 ? 2.6 : 2.0)
    end
    return p
end

function save_figures(cfg,runs,traces,grid,surface,reference)
    out=cfg.output
    p=line_figure(runs,traces;ylabel="Objective value J(z)",
      title="Softmax sensitivity: original hard-max objective\nOne independent common CRN bank, $(cfg.trace_paths) paths")
    savefig(p,joinpath(out,"softmax_sensitivity_convergence.png"))
    if reference!==nothing
        for (index,color,label) in ((1,:gray55,"Fixed (reference)"),(3,:darkcyan,"Green (reference)"))
            plot!(p,0:length(reference.traces[index])-1,reference.traces[index];label,color,lw=2,linestyle=:dashdot)
        end
    end
    savefig(p,joinpath(out,"softmax_sensitivity_all_methods.png"))
    for (field,file,ylabel,title) in (
        (:K_eff,"softmax_effective_policies.png","K_eff","Effective policies along each Softmax trajectory"),
        (:relative_difference,"softmax_gradient_difference.png","Relative gradient difference","Softmax versus hard argmax at the SAME point and bank"),
        (:cosine,"softmax_gradient_cosine.png","Cosine similarity","Gradient directions at the SAME point and bank"))
        series=[[ismissing(getproperty(d,field)) ? NaN : getproperty(d,field) for d in r.diagnostics] for r in runs]
        q=line_figure(runs,series;ylabel,title,omit_fresh=true)
        field==:cosine && ylims!(q,-1.05,1.05)
        savefig(q,joinpath(out,file))
    end
    coords=map(1:2) do j
        line_figure(runs,[[z[j] for z in r.zhist] for r in runs];ylabel=j==1 ? "z3" : "z4",title=j==1 ? "Class-3 probing" : "Class-4 probing")
    end
    p=plot(coords...;layout=(1,2),size=(1500,650),legendfontsize=7)
    savefig(p,joinpath(out,"softmax_probing_coordinates.png"))
    p=contourf(grid,grid,surface;levels=24,xlabel="z3",ylabel="z4",colorbar_title="Oracle J(z)",
       title="Fresh and Softmax trajectories on the cached oracle surface",size=(1200,900),
       left_margin=10Plots.mm,bottom_margin=10Plots.mm,right_margin=8Plots.mm,titlefontsize=12,legendfontsize=8)
    for (i,r) in enumerate(runs)
        color=COLORS[mod1(i,length(COLORS))]
        plot!(p,[z[1] for z in r.zhist],[z[2] for z in r.zhist];label=r.method,color,
              linestyle=STYLES[mod1(i,length(STYLES))],lw=1.8)
        scatter!(p,[r.zhist[end][1]],[r.zhist[end][2]];color,ms=5,label="")
    end
    scatter!(p,[C.Z0[1]],[C.Z0[2]];marker=:star5,color=:gold,ms=9,label="Common z0")
    savefig(p,joinpath(out,"softmax_trajectories_oracle.png"))
    panels=map((:gradient_difference,:clipped_gradient_difference,:projected_update_difference)) do field
        line_figure(runs,[[getproperty(d,field) for d in r.diagnostics] for r in runs];omit_fresh=true,
            ylabel=string(field),title=field==:gradient_difference ? "Before clipping" : field==:clipped_gradient_difference ? "After clipping" : "After projection (update gap)")
    end
    p=plot(panels...;layout=(1,3),size=(1800,650),legendfontsize=6)
    savefig(p,joinpath(out,"softmax_update_effects.png"))
end

function snapshot_originals()
    paths=[joinpath(@__DIR__,f) for f in ("002probing_three_searches_n500_full45iters.jl","003probing_four_searches_softmax.jl","003probing_four_searches_softmax_README.txt")]
    for dir in ("resultados/002probing_20261021","resultados/003probing_softmax_20261021")
        for (root,_,files) in walkdir(joinpath(@__DIR__,dir)),f in files;push!(paths,joinpath(root,f));end
    end
    Dict(relpath(p,@__DIR__)=>bytes2hex(sha256(read(p))) for p in paths)
end

function run_experiment(cfg)
    start=time();mkpath(cfg.output)
    original_hashes=snapshot_originals()
    @assert !(cfg.final_seed in [cfg.search_seed+10000*k for k in 1:cfg.iterations])
    @assert cfg.final_seed!=cfg.trace_seed && cfg.final_seed!=cfg.surface_seed
    policies=C.build_policy_grid();@assert length(policies)==144
    println("Fresh + six-beta sensitivity; ex-ante hard-max objective unchanged.")
    println("Configuration: ",cfg);flush(stdout)
    reference=load_reference(cfg)
    runs,anchors,unique_banks=run_searches(cfg,policies;reference)
    serialize(joinpath(cfg.output,"search_checkpoint.jls"),(runs=runs,anchors=anchors,config=cfg))
    diagnostics=[merge((method=r.method,beta=r.beta),d) for r in runs for d in r.diagnostics]
    write_csv(joinpath(cfg.output,"iteration_diagnostics.csv"),diagnostics)
    write_csv(joinpath(cfg.output,"fixed_point_beta_diagnostics.csv"),anchors)
    traces,newpoints=evaluate_traces(cfg,runs,policies,reference)
    grid,surface,surface_reused=reference_surface(cfg,policies,reference)
    oracle,pairs,selected_rewards=evaluate_endpoints(cfg,runs,policies)
    aggregates=aggregate_results(cfg,runs,traces,oracle)
    write_csv(joinpath(cfg.output,"aggregate_indicators.csv"),aggregates)
    write_csv(joinpath(cfg.output,"final_evaluation.csv"),oracle)
    write_csv(joinpath(cfg.output,"paired_final_differences.csv"),pairs)
    tracerows=[(method=r.method,beta=r.beta,iteration=k-1,z3=r.zhist[k][1],z4=r.zhist[k][2],
                J_common_crn=traces[i][k]) for (i,r) in enumerate(runs) for k in eachindex(r.zhist)]
    write_csv(joinpath(cfg.output,"convergence_traces.csv"),tracerows)
    serialize(joinpath(cfg.output,"sensitivity_results.jls"),(runs=runs,traces=traces,anchors=anchors,
              aggregates=aggregates,oracle=oracle,pairs=pairs,selected_rewards=selected_rewards,
              grid=grid,surface=surface,config=cfg))
    save_figures(cfg,runs,traces,grid,surface,reference)
    @assert snapshot_originals()==original_hashes
    write_csv(joinpath(cfg.output,"original_files_sha256.csv"),[(path=p,sha256=h) for (p,h) in sort(collect(original_hashes))])
    metadata=Dict("created_utc"=>string(Dates.now(Dates.UTC)),"elapsed_seconds"=>time()-start,
      "source_sha256"=>bytes2hex(sha256(read(@__FILE__))),"core_source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"003probing_four_searches_softmax.jl")))),
      "betas"=>cfg.betas,"N_PATHS"=>cfg.npaths,"JOBS_PER_PATH"=>cfg.njobs,"MAX_ITERS"=>cfg.iterations,
      "Z0"=>C.Z0,"ALPHA"=>C.ALPHA,"BETA"=>C.BETA,"BASE_STEP"=>C.BASE_STEP,"GRAD_CLIP"=>C.GRAD_CLIP,
      "policy_count"=>length(policies),"TRACE_EVAL_PATHS"=>cfg.trace_paths,"TRACE_EVAL_SEED"=>cfg.trace_seed,
      "FINAL_EVAL_PATHS"=>cfg.final_paths,"FINAL_EVAL_SEED"=>cfg.final_seed,"SEARCH_SEED_FRESH"=>cfg.search_seed,
      "iteration_seed_stride"=>10000,"epsilon"=>cfg.epsilon,"surface_reused"=>surface_reused,
      "references_directory"=>cfg.references,"reused_searches"=>[r.method for r in runs if r.reused],
      "unique_local_point_banks"=>unique_banks,"new_trace_points_evaluated"=>newpoints,
      "logical_search_paths_per_method"=>cfg.npaths*cfg.iterations,
      "primitive_search_paths_generated"=>cfg.npaths*cfg.iterations,
      "bank_replay_note"=>"Saved Fresh/0.5 decisions are reused; their banks are reconstructed only to obtain missing same-bank diagnostics.",
      "objective"=>"max_i E_z[Y_i]-alpha'z; no softening of reported objective",
      "oracle_note"=>"One NEW common CRN bank independent of searches, traces and surface. SE/paired CI do not adjust for policy selection or multiplicity.",
      "julia_version"=>string(VERSION),"plots_version"=>string(pkgversion(Plots)),
      "blas_threads"=>BLAS.get_num_threads(),"original_files_unchanged"=>true)
    open(joinpath(cfg.output,"parameters.toml"),"w") do io;TOML.print(io,metadata);end
    println("\nFinal descriptive comparison:")
    for a in aggregates
        @printf("%s | J_CRN=%.6f | J_final=%.6f SE=%.6f | K_eff=%.3f | D=%.3f | TV=%.3f\n",
                a.method,a.J_final,a.oracle_objective,a.oracle_SE,a.K_eff_mean,a.relative_difference_mean,a.total_variation)
    end
    @printf("Elapsed %.2f s; originals unchanged. Results: %s\n",time()-start,cfg.output);flush(stdout)
    return (runs=runs,traces=traces,anchors=anchors,aggregates=aggregates,oracle=oracle,pairs=pairs)
end

function main(args=ARGS)
    cfg=parse_arguments(args)
    cfg===nothing && return
    run_experiment(cfg)
end
if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
