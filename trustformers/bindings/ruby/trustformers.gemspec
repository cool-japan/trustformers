# frozen_string_literal: true

require_relative "lib/trustformers/version"

Gem::Specification.new do |spec|
  spec.name = "trustformers"
  spec.version = TrustFormeRS::VERSION
  spec.authors = ["TrustFormeRS Team"]
  spec.email = ["contact@trustformers.ai"]

  spec.summary = "Ruby bindings for TrustFormeRS - High-performance transformer models"
  spec.description = <<~DESC
    TrustFormeRS is a high-performance Rust library for transformer models with Ruby bindings.
    It provides efficient implementations of popular transformer architectures including BERT,
    GPT, T5, and more, with support for text generation, classification, question answering,
    and other NLP tasks.
  DESC
  spec.homepage = "https://github.com/cool-japan/trustformers"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.0.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/cool-japan/trustformers"
  spec.metadata["changelog_uri"] = "https://github.com/cool-japan/trustformers/blob/main/CHANGELOG.md"
  spec.metadata["documentation_uri"] = "https://docs.trustformers.ai/ruby"
  spec.metadata["bug_tracker_uri"] = "https://github.com/cool-japan/trustformers/issues"

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (f == __FILE__) ||
        f.match(%r{\A(?:(?:bin|test|spec|features)/|\.(?:git|travis|circleci)|appveyor)}) ||
        f.start_with?("examples/") # Exclude examples from gem
    end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/trustformers/extconf.rb"]

  # Runtime dependencies
  spec.add_dependency "ffi", "~> 1.15"

  # Development dependencies
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.2"
  spec.add_development_dependency "minitest", "~> 5.0"
  spec.add_development_dependency "yard", "~> 0.9"
  spec.add_development_dependency "rubocop", "~> 1.21"
  spec.add_development_dependency "rubocop-minitest", "~> 0.25"
  spec.add_development_dependency "rubocop-rake", "~> 0.6"
  spec.add_development_dependency "benchmark-ips", "~> 2.10"

  # Platform-specific dependencies
  spec.platform = Gem::Platform::RUBY

  # Metadata for gem publication
  spec.metadata["rubygems_mfa_required"] = "true"
end