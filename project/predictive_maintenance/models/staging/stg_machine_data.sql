with source as (

    select * from {{ source('raw', 'raw_sensor_data') }}

),

renamed as (

    select
        machine_id,
        timestamp as recorded_at,
        temperature,
        vibration,
        humidity,
        pressure,
        energy_consumption,
        machine_status,
        anomaly_flag,
        predicted_remaining_life,
        failure_type,
        downtime_risk,
        maintenance_required

    from source

)

select * from renamed
