//! The ZMQ sender binds hostnames like pyzmq's libzmq (the default host is `localhost`).

use prl_config::{TransportConfig, ZMQTransportConfig};
use prl_train::BatchSender;

#[tokio::test]
async fn zmq_sender_binds_a_hostname() {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let transport = TransportConfig::Zmq(ZMQTransportConfig { host: "localhost".into(), port });
    let sender = BatchSender::new("/tmp".into(), 1, 0, &transport).await.unwrap();
    sender.close().await;
}
