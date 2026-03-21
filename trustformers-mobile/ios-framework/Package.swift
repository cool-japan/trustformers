// swift-tools-version:5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TrustformersKit",
    platforms: [
        .iOS(.v11),
        .macOS(.v10_13),
        .watchOS(.v4),
        .tvOS(.v11)
    ],
    products: [
        // Main library product
        .library(
            name: "TrustformersKit",
            targets: ["TrustformersKit", "TrustformersKitCore"]),
        
        // Core-only product (minimal dependencies)
        .library(
            name: "TrustformersKitCore",
            targets: ["TrustformersKitCore"]),
            
        // Performance monitoring product
        .library(
            name: "TrustformersKitPerformance",
            targets: ["TrustformersKit"]),
            
        // Privacy features product
        .library(
            name: "TrustformersKitPrivacy", 
            targets: ["TrustformersKit"]),
            
        // AR integration product
        .library(
            name: "TrustformersKitAR",
            targets: ["TrustformersKit"]),
    ],
    dependencies: [
        // No external dependencies to maintain minimal footprint
    ],
    targets: [
        // Binary target for the Rust library
        .binaryTarget(
            name: "TrustformersKitCore",
            path: "TrustformersKit.xcframework"
        ),
        
        // Swift wrapper target
        .target(
            name: "TrustformersKit",
            dependencies: ["TrustformersKitCore"],
            path: "TrustformersKit/Sources",
            publicHeadersPath: "../Headers",
            cSettings: [
                .headerSearchPath("../Headers"),
                .define("TRUSTFORMERS_KIT_VERSION", to: "\"1.0.0\""),
                .define("TRUSTFORMERS_SPM", to: "1"),
                .unsafeFlags(["-fmodules", "-fcxx-modules"], .when(platforms: [.iOS, .macOS]))
            ],
            swiftSettings: [
                .define("TRUSTFORMERS_SPM"),
                .define("TRUSTFORMERS_MOBILE", .when(platforms: [.iOS, .watchOS, .tvOS])),
                .unsafeFlags(["-O", "-whole-module-optimization"], .when(configuration: .release))
            ]
        ),
        
        // Test target
        .testTarget(
            name: "TrustformersKitTests",
            dependencies: ["TrustformersKit"],
            path: "Tests",
            swiftSettings: [
                .define("TRUSTFORMERS_TESTING")
            ]
        ),
    ],
    swiftLanguageVersions: [.v5]
)