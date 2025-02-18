use std::{io, io::Write, process::Command};

fn main() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let assets_dir = std::path::PathBuf::from(&manifest_dir)
        .join("assets")
        .join("tailwind");
    std::fs::create_dir_all(&assets_dir).expect("Failed to create assets directory");

    let tailwind_binary_name = if cfg!(target_os = "linux") {
        "tailwindcss-linux-x64"
    } else if cfg!(target_os = "macos") {
        match std::env::consts::ARCH {
            "aarch64" => "tailwindcss-macos-arm64",
            "x86_64" => "tailwindcss-macos-x64",
            arch => panic!("Unsupported architecture on macOS: {}", arch),
        }
    } else if cfg!(target_os = "windows") {
        "tailwindcss-windows-x64.exe"
    } else {
        panic!("Unsupported operating system")
    };
    let tailwind_binary_path = std::path::PathBuf::from(&assets_dir).join(tailwind_binary_name);

    if !tailwind_binary_path.exists() {
        println!("cargo:warning=Downloading Tailwind CSS binary...");

        let url = format!(
            "https://github.com/tailwindlabs/tailwindcss/releases/latest/download/{}",
            tailwind_binary_name
        );

        match Command::new("curl")
            .args(["-sL", "-o", tailwind_binary_path.to_str().unwrap(), &url])
            .output()
        {
            Ok(output) => {
                if !output.status.success() {
                    let _ = io::stdout().write_all(&output.stdout);
                    let _ = io::stdout().write_all(&output.stderr);
                    panic!("Failed to download Tailwind binary");
                }
            }
            Err(e) => panic!("Tailwind error: {:?}", e),
        };

        #[cfg(unix)]
        std::fs::set_permissions(
            &tailwind_binary_path,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .expect("Failed to set executable permissions");
    }

    let daisyui_css_filename = "daisyui.css";
    let daisyui_css_path = std::path::PathBuf::from(&assets_dir).join(daisyui_css_filename);
    if !daisyui_css_path.exists() {
        println!("cargo:warning=Downloading DaisyUI CSS file...");

        let url = "https://cdn.jsdelivr.net/combine/npm/daisyui@beta/base/properties.css,npm/daisyui@beta/base/reset.css,npm/daisyui@beta/base/rootcolor.css,npm/daisyui@beta/base/rootscrollgutter.css,npm/daisyui@beta/base/rootscrolllock.css,npm/daisyui@beta/base/scrollbar.css,npm/daisyui@beta/base/svg.css,npm/daisyui@beta/components/alert.css,npm/daisyui@beta/components/avatar.css,npm/daisyui@beta/components/badge.css,npm/daisyui@beta/components/breadcrumbs.css,npm/daisyui@beta/components/button.css,npm/daisyui@beta/components/calendar.css,npm/daisyui@beta/components/card.css,npm/daisyui@beta/components/carousel.css,npm/daisyui@beta/components/chat.css,npm/daisyui@beta/components/checkbox.css,npm/daisyui@beta/components/collapse.css,npm/daisyui@beta/components/countdown.css,npm/daisyui@beta/components/diff.css,npm/daisyui@beta/components/divider.css,npm/daisyui@beta/components/dock.css,npm/daisyui@beta/components/drawer.css,npm/daisyui@beta/components/dropdown.css,npm/daisyui@beta/components/fieldset.css,npm/daisyui@beta/components/fileinput.css,npm/daisyui@beta/components/filter.css,npm/daisyui@beta/components/footer.css,npm/daisyui@beta/components/hero.css,npm/daisyui@beta/components/indicator.css,npm/daisyui@beta/components/input.css,npm/daisyui@beta/components/kbd.css,npm/daisyui@beta/components/label.css,npm/daisyui@beta/components/link.css,npm/daisyui@beta/components/list.css,npm/daisyui@beta/components/loading.css,npm/daisyui@beta/components/mask.css,npm/daisyui@beta/components/menu.css,npm/daisyui@beta/components/mockup.css,npm/daisyui@beta/components/modal.css,npm/daisyui@beta/components/navbar.css,npm/daisyui@beta/components/progress.css,npm/daisyui@beta/components/radialprogress.css,npm/daisyui@beta/components/radio.css,npm/daisyui@beta/components/range.css,npm/daisyui@beta/components/rating.css,npm/daisyui@beta/components/select.css,npm/daisyui@beta/components/skeleton.css,npm/daisyui@beta/components/stack.css,npm/daisyui@beta/components/stat.css,npm/daisyui@beta/components/status.css,npm/daisyui@beta/components/steps.css,npm/daisyui@beta/components/swap.css,npm/daisyui@beta/components/tab.css,npm/daisyui@beta/components/table.css,npm/daisyui@beta/components/textarea.css,npm/daisyui@beta/components/timeline.css,npm/daisyui@beta/components/toast.css,npm/daisyui@beta/components/toggle.css,npm/daisyui@beta/components/tooltip.css,npm/daisyui@beta/components/validator.css,npm/daisyui@beta/utilities/glass.css,npm/daisyui@beta/utilities/join.css,npm/daisyui@beta/utilities/radius.css,npm/daisyui@beta/utilities/typography.css,npm/daisyui@beta/colors/properties.css,npm/daisyui@beta/colors/responsive.css,npm/daisyui@beta/colors/states.css";

        match Command::new("curl")
            .args(["-sL", "-o", daisyui_css_path.to_str().unwrap(), &url])
            .output()
        {
            Ok(output) => {
                if !output.status.success() {
                    let _ = io::stdout().write_all(&output.stdout);
                    let _ = io::stdout().write_all(&output.stderr);
                    panic!("Failed to download DaisyUI CSS file");
                }
            }
            Err(e) => panic!("Tailwind error: {:?}", e),
        };
        #[cfg(unix)]
        std::fs::set_permissions(
            daisyui_css_filename,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .expect("Failed to set executable permissions");
    }

    println!("cargo:warning=Building Tailwind CSS file to ./assets/tailwind/tailwindcss...");

    // Build Tailwind CSS
    match Command::new(tailwind_binary_path)
        .args(["-i", "./assets/input.css", "-o", "./public/tailwind.css"])
        .output()
    {
        Ok(output) => {
            if !output.status.success() {
                let _ = io::stdout().write_all(&output.stdout);
                let _ = io::stdout().write_all(&output.stderr);
                panic!("Tailwind error");
            }
        }
        Err(e) => panic!("Tailwind error: {:?}", e),
    };

    println!("cargo:rerun-if-changed=./assets/*");
    println!("cargo:rerun-if-changed=./src/**/*");
}
