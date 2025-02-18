FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        clang \
        pkg-config \
        libssl-dev \
        sudo \
        tzdata \
        curl \
        g++ \
        protobuf-compiler \
        llvm-dev \
        libclang-dev \
        sqlite3 \
        cmake

# Set the timezone environment variable
ENV TZ=UTC

# Configure timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt install -y git-lfs

# Set the working directory
WORKDIR /root

# Install Rust as the root user
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . "/root/.cargo/env" && \
    rustup toolchain install nightly-x86_64-unknown-linux-gnu && \
    rustup target add wasm32-unknown-unknown && \
    rustup component add rustfmt --toolchain nightly-x86_64-unknown-linux-gnu && \
    rustup component add clippy --toolchain nightly-x86_64-unknown-linux-gnu && \
    rustup default nightly-x86_64-unknown-linux-gnu

# Set PATH for all future RUN commands and for containers started from this image
ENV PATH="/root/.cargo/bin:$PATH"

# Dioxus
run curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash 

RUN cargo binstall -y dioxus-cli





