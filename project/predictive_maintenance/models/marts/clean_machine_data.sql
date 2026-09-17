with staged as (

    select * from {{ ref('stg_machine_data') }}

),

cleaned as (

    select *
    from staged

    -- Suppression des observations présentant une vibration négative.
    -- Analyse exploratoire : 37 lignes sur 100 000 (0,04 %), dont 6 lignes
    -- avec maintenance_required = 1 (soit seulement 0,03 % des 19 697 cas
    -- positifs du dataset — impact négligeable sur le volume disponible pour le ML).
    -- Ces valeurs étant physiquement impossibles (une vibration ne peut pas
    -- être négative) et leur proportion étant négligeable, les observations
    -- sont supprimées afin de garantir la qualité des données, plutôt que
    -- d'inventer une correction (ex: valeur absolue) qui ne serait pas justifiable.
    where vibration >= 0

)

select * from cleaned
