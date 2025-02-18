use super::{
    health::{health_request, HealthStatus},
    models::{model_request, ModelStatus},
};
use crate::llms::{api::ApiClient, local::llama_cpp::LlamaCppConfig};
use tokio::time::{sleep, timeout, Duration, Instant};

#[derive(PartialEq)]
pub(super) enum ServerStatus {
    RunningModel(String),
    RunningRequested,
    Offline,
}

pub(super) async fn server_status(
    requested_model_path: &str,
    server_http_path: &str,
    test_time: std::time::Duration,
    retry_time: std::time::Duration,
    client: &ApiClient<LlamaCppConfig>,
) -> crate::Result<ServerStatus> {
    let start_time = Instant::now();
    // First, test the TCP connection
    if let Err(e) = timeout(test_time, test_connection(server_http_path, retry_time)).await {
        crate::trace!(
            "TCP connection to {} failed after {:?}: {}",
            server_http_path,
            test_time,
            e
        );
        return Ok(ServerStatus::Offline);
    }

    // Then, repeatedly check the health status
    loop {
        if Instant::now().duration_since(start_time) >= test_time {
            crate::bail!(
                "Health check for {} failed after {:?}",
                server_http_path,
                test_time
            );
        }

        match health_request(client).await {
            HealthStatus::Alive => break,
            // HealthStatus::Loading => {
            //     sleep(retry_time).await;
            // }
            HealthStatus::ErrorOrOffline(e) => {
                crate::trace!(
                    "Health check for failed ({e}). Retrying after: {}ms",
                    retry_time.as_millis()
                );
                sleep(retry_time).await;
            }
        }
    }

    match model_request(client).await {
        Ok(ModelStatus::LoadedModel(model_id)) => {
            if &requested_model_path == &model_id {
                return Ok(ServerStatus::RunningRequested);
            } else {
                crate::info!(
                    "Model {} is loaded, but requested model is {}",
                    model_id,
                    &requested_model_path
                );
                return Ok(ServerStatus::RunningModel(model_id));
            }
        }
        Ok(ModelStatus::_LoadedModels(_model_ids)) => {
            todo!()
        }
        Err(e) => {
            crate::bail!("Model check for {} failed: {}", server_http_path, e);
        }
    }
}

pub(crate) async fn test_connection(
    server_http_path: &str,
    retry_time: Duration,
) -> crate::Result<()> {
    loop {
        match tokio::net::TcpStream::connect(server_http_path).await {
            Ok(_) => return Ok(()),
            Err(_) => {
                sleep(retry_time).await;
            }
        }
    }
}
