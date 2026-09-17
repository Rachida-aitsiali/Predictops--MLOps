{{ config(materialized='table') }}
with cleaned as (

    select * from {{ ref('clean_machine_data') }}

),

final as (

    select
        machine_id,
        recorded_at,
        temperature,
        vibration,
        humidity,
        pressure,
        energy_consumption,
        machine_status,
        anomaly_flag,
        failure_type,

        -- Score de risque d'arrêt (0 à 1). À noter : quasi-redondant avec
        -- anomaly_flag d'après l'analyse exploratoire (8912/8914 valeurs à 1.0
        -- correspondent exactement aux cas anomaly_flag = 1). Conservé ici
        -- pour laisser le choix à l'équipe ML de l'utiliser ou non.
        downtime_risk,

        -- Durée de vie restante estimée. À noter : l'analyse exploratoire
        -- montre une moyenne quasi identique (233-235) quelle que soit la
        -- valeur de machine_status, suggérant une faible valeur prédictive
        -- réelle pour cette variable dans ce dataset.
        predicted_remaining_life,

        -- Variable cible pour le Machine Learning
        maintenance_required

    from cleaned

)

select * from final