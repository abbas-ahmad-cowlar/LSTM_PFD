"""
Role-Based Access Control (RBAC) Permissions Model.

Reference: Master Roadmap Chapter 4.1.2

This module implements a role-based permission system with four roles:
- VIEWER: Read-only access to experiments and results
- OPERATOR: Can run experiments, upload datasets, use XAI
- ADMIN: Full access plus user management
- SUPER_ADMIN: System configuration access

Usage:
    from packages.dashboard.models.permissions import Role, has_permission, require_permission
    
    # Check permission
    if has_permission(current_user, 'create_experiments'):
        # Allow action
    
    # As decorator
    @require_permission('manage_users')
    def admin_view():
        pass
"""

from enum import Enum
from functools import wraps
from typing import List, Optional, Set


class Role(Enum):
    """User roles with increasing privilege levels."""
    VIEWER = "viewer"           # Read-only access
    OPERATOR = "operator"       # Run experiments, view results
    ADMIN = "admin"             # Full access except system config
    SUPER_ADMIN = "super_admin" # Full system access


# Permission definitions per role
ROLE_PERMISSIONS: dict[Role, Set[str]] = {
    Role.VIEWER: {
        'view_experiments',
        'view_datasets',
        'view_results',
        'view_models',
        'view_xai_explanations',
    },
    Role.OPERATOR: {
        # Inherit VIEWER permissions
        'view_experiments',
        'view_datasets',
        'view_results',
        'view_models',
        'view_xai_explanations',
        # Additional permissions
        'create_experiments',
        'run_experiments',
        'stop_experiments',
        'upload_datasets',
        'delete_own_datasets',
        'export_results',
        'use_xai_dashboard',
        'run_hpo_campaigns',
        'view_hpo_results',
        'create_model_versions',
    },
    Role.ADMIN: {
        # Wildcard for most permissions
        '*',
        # Explicitly listed for clarity
        'manage_users',
        'manage_api_keys',
        'view_system_health',
        'configure_notifications',
        'delete_any_dataset',
        'delete_any_experiment',
        'view_audit_logs',
        'manage_model_deployments',
    },
    Role.SUPER_ADMIN: {
        # Full system access
        '**',
        'system_config',
        'database_admin',
        'security_settings',
        'backup_restore',
        'delete_all',
    },
}

# Permissions that require SUPER_ADMIN even with '*' wildcard
SUPER_ADMIN_ONLY = {
    'system_config',
    'database_admin',
    'security_settings',
    'backup_restore',
    'delete_all',
}


def get_role_permissions(role: Role) -> Set[str]:
    """
    Get all permissions for a role.
    
    Args:
        role: The user's role
        
    Returns:
        Set of permission strings
    """
    return ROLE_PERMISSIONS.get(role, set())


def has_permission(user, permission: str) -> bool:
    """
    Check if a user has a specific permission.
    
    Args:
        user: User object with a 'role' attribute (Role enum or string)
        permission: Permission string to check
        
    Returns:
        True if user has the permission, False otherwise
    """
    # Handle role as string or enum
    if isinstance(user.role, str):
        try:
            role = Role(user.role)
        except ValueError:
            return False
    else:
        role = user.role
    
    role_perms = ROLE_PERMISSIONS.get(role, set())
    
    # Super admin has all permissions
    if '**' in role_perms:
        return True
    
    # Admin wildcard covers most permissions except SUPER_ADMIN_ONLY
    if '*' in role_perms:
        if permission not in SUPER_ADMIN_ONLY:
            return True
    
    # Direct permission check
    return permission in role_perms


def has_any_permission(user, permissions: List[str]) -> bool:
    """
    Check if user has any of the specified permissions.
    
    Args:
        user: User object with a 'role' attribute
        permissions: List of permission strings
        
    Returns:
        True if user has at least one permission
    """
    return any(has_permission(user, p) for p in permissions)


def has_all_permissions(user, permissions: List[str]) -> bool:
    """
    Check if user has all of the specified permissions.
    
    Args:
        user: User object with a 'role' attribute
        permissions: List of permission strings
        
    Returns:
        True if user has all permissions
    """
    return all(has_permission(user, p) for p in permissions)


def require_permission(permission: str):
    """
    Decorator to require a specific permission for a view/function.
    
    Usage:
        @require_permission('manage_users')
        def admin_only_view():
            pass
    
    Args:
        permission: Required permission string
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import here to avoid circular imports
            from flask import g, abort
            
            user = getattr(g, 'current_user', None)
            if user is None:
                abort(401, description="Authentication required")
            
            if not has_permission(user, permission):
                abort(403, description=f"Permission denied: {permission}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_any_permission(*permissions: str):
    """
    Decorator requiring any of the specified permissions.
    
    Usage:
        @require_any_permission('view_experiments', 'create_experiments')
        def experiment_view():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import g, abort
            
            user = getattr(g, 'current_user', None)
            if user is None:
                abort(401, description="Authentication required")
            
            if not has_any_permission(user, list(permissions)):
                abort(403, description=f"Permission denied: requires one of {permissions}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(min_role: Role):
    """
    Decorator requiring a minimum role level.
    
    Role hierarchy: VIEWER < OPERATOR < ADMIN < SUPER_ADMIN
    
    Usage:
        @require_role(Role.ADMIN)
        def admin_view():
            pass
    """
    role_hierarchy = [Role.VIEWER, Role.OPERATOR, Role.ADMIN, Role.SUPER_ADMIN]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import g, abort
            
            user = getattr(g, 'current_user', None)
            if user is None:
                abort(401, description="Authentication required")
            
            # Get user's role
            if isinstance(user.role, str):
                try:
                    user_role = Role(user.role)
                except ValueError:
                    abort(403, description="Invalid role")
                    return
            else:
                user_role = user.role
            
            # Check role hierarchy
            if role_hierarchy.index(user_role) < role_hierarchy.index(min_role):
                abort(403, description=f"Minimum role required: {min_role.value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience decorators
viewer_required = require_role(Role.VIEWER)
operator_required = require_role(Role.OPERATOR)
admin_required = require_role(Role.ADMIN)
super_admin_required = require_role(Role.SUPER_ADMIN)


def get_user_permissions(user) -> Set[str]:
    """
    Get all explicit permissions for a user.
    
    Args:
        user: User object with a 'role' attribute
        
    Returns:
        Set of permission strings (does not expand wildcards)
    """
    if isinstance(user.role, str):
        try:
            role = Role(user.role)
        except ValueError:
            return set()
    else:
        role = user.role
    
    return ROLE_PERMISSIONS.get(role, set()).copy()


if __name__ == '__main__':
    # Test the permission system
    class MockUser:
        def __init__(self, role):
            self.role = role
    
    # Test each role
    viewer = MockUser(Role.VIEWER)
    operator = MockUser(Role.OPERATOR)
    admin = MockUser(Role.ADMIN)
    super_admin = MockUser(Role.SUPER_ADMIN)
    
    print("Permission Tests:")
    print(f"  VIEWER can view_experiments: {has_permission(viewer, 'view_experiments')}")  # True
    print(f"  VIEWER can create_experiments: {has_permission(viewer, 'create_experiments')}")  # False
    print(f"  OPERATOR can create_experiments: {has_permission(operator, 'create_experiments')}")  # True
    print(f"  ADMIN can manage_users: {has_permission(admin, 'manage_users')}")  # True
    print(f"  ADMIN can system_config: {has_permission(admin, 'system_config')}")  # False
    print(f"  SUPER_ADMIN can system_config: {has_permission(super_admin, 'system_config')}")  # True
