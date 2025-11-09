// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TrustformeRS",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .watchOS(.v8),
        .tvOS(.v15)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "TrustformeRS",
            targets: ["TrustformeRS"]),
        .library(
            name: "TrustformeRSCore",
            targets: ["TrustformeRSCore"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        .target(
            name: "TrustformeRSCore",
            dependencies: [],
            path: "Sources/TrustformeRSCore",
            cSettings: [
                .headerSearchPath("include"),
            ],
            linkerSettings: [
                .linkedLibrary("trustformers"),
                .linkedFramework("Accelerate"),
                .linkedFramework("MetalKit", .when(platforms: [.macOS, .iOS])),
                .linkedFramework("CoreML"),
            ]
        ),
        .target(
            name: "TrustformeRS",
            dependencies: ["TrustformeRSCore"],
            path: "Sources/TrustformeRS",
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals"),
                .enableUpcomingFeature("ConciseMagicFile"),
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("ForwardTrailingClosures"),
                .enableUpcomingFeature("ImportObjcForwardDeclarations"),
                .enableUpcomingFeature("DisableOutwardActorInference"),
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
        .testTarget(
            name: "TrustformeRSTests",
            dependencies: ["TrustformeRS"],
            path: "Tests/TrustformeRSTests"),
        .testTarget(
            name: "TrustformeRSCoreTests",
            dependencies: ["TrustformeRSCore"],
            path: "Tests/TrustformeRSCoreTests"),
    ],
    cLanguageStandard: .c17,
    cxxLanguageStandard: .cxx20
)