using ContinuationSuite
using Documenter

DocMeta.setdocmeta!(ContinuationSuite, :DocTestSetup, :(using ContinuationSuite); recursive=true)

makedocs(;
    modules=[ContinuationSuite],
    authors="Javier González Monge",
    repo="https://github.com/RayleighLord/ContinuationSuite.jl/blob/{commit}{path}#{line}",
    sitename="ContinuationSuite.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RayleighLord.github.io/ContinuationSuite.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RayleighLord/ContinuationSuite.jl",
    devbranch="main",
)
