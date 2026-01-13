"""
Test all Phase 1 layouts to verify they can be imported.
"""
import os
import sys

# Set required environment variables
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['SECRET_KEY'] = 'test_secret_key_12345'
os.environ['SKIP_CONFIG_VALIDATION'] = 'True'

# Add dashboard to path
sys.path.insert(0, 'packages/dashboard')

# Phase 1 layouts to test (per MASTER_ROADMAP_FINAL.md)
PHASE_1_LAYOUTS = {
    'home': 'create_home_layout',
    'datasets': 'create_datasets_layout',
    'hpo_campaigns': 'create_hpo_campaigns_layout',
    'deployment': 'create_deployment_layout',
    'system_health': 'create_system_health_layout',
    'api_monitoring': 'create_api_monitoring_layout',
    'evaluation_dashboard': 'create_evaluation_dashboard_layout',
    'testing_dashboard': 'create_testing_dashboard_layout',
    'feature_engineering': 'create_feature_engineering_layout',
    'xai_dashboard': 'create_xai_dashboard_layout',
    'nas_dashboard': 'create_nas_dashboard_layout',
    'data_explorer': 'create_data_explorer_layout',
    'experiments': 'create_experiments_layout',
    'visualization': 'create_visualization_layout',
    'settings': 'create_settings_layout',
}

def test_layouts():
    """Test importing all Phase 1 layouts."""
    results = {'passed': [], 'failed': []}
    
    for module, func in PHASE_1_LAYOUTS.items():
        try:
            layout_module = __import__(f'layouts.{module}', fromlist=[func])
            create_func = getattr(layout_module, func)
            print(f"  ✓ {module}")
            results['passed'].append(module)
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            results['failed'].append((module, str(e)))
    
    return results

def test_callbacks():
    """Test importing callback registration."""
    print("\nTesting callbacks/__init__.py...")
    try:
        from callbacks import register_all_callbacks
        print("  ✓ register_all_callbacks imported")
        return True
    except Exception as e:
        print(f"  ✗ Callbacks import failed: {e}")
        return False

def test_services():
    """Test importing key services."""
    print("\nTesting key services...")
    services = [
        'hpo_service',
        'deployment_service', 
        'dataset_service',
        'evaluation_service',
        'testing_service',
        'feature_service',
        'xai_service',
    ]
    results = {'passed': [], 'failed': []}
    
    for service in services:
        try:
            __import__(f'services.{service}')
            print(f"  ✓ {service}")
            results['passed'].append(service)
        except Exception as e:
            print(f"  ✗ {service}: {e}")
            results['failed'].append((service, str(e)))
    
    return results

if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 1 VERIFICATION TEST")
    print("=" * 60)
    
    print("\nTesting Phase 1 layouts...")
    layout_results = test_layouts()
    
    callbacks_ok = test_callbacks()
    
    service_results = test_services()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nLayouts:   {len(layout_results['passed'])}/{len(PHASE_1_LAYOUTS)} passed")
    print(f"Callbacks: {'OK' if callbacks_ok else 'FAILED'}")
    print(f"Services:  {len(service_results['passed'])}/7 passed")
    
    if layout_results['failed']:
        print("\n⚠️  Failed layouts:")
        for module, error in layout_results['failed']:
            print(f"   - {module}: {error[:80]}...")
    
    if service_results['failed']:
        print("\n⚠️  Failed services:")
        for service, error in service_results['failed']:
            print(f"   - {service}: {error[:80]}...")
    
    total_failed = len(layout_results['failed']) + len(service_results['failed']) + (0 if callbacks_ok else 1)
    
    if total_failed == 0:
        print("\n✅ ALL PHASE 1 COMPONENTS VERIFIED!")
        sys.exit(0)
    else:
        print(f"\n❌ {total_failed} components need fixing")
        sys.exit(1)
