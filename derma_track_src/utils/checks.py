from django.shortcuts import redirect
from functools import wraps

def group_and_super_user_checks(group_names, redirect_url='/'):
    """
    Decorator to restrict access to users in specific groups or super user, redirecting unauthorized users.

    Args:
        group_names (str or list): The group name(s) to check. Can be a string or a list of strings.
        redirect_url (str): The URL name to redirect unauthorized users. Defaults to 'core:index'.

    Returns:
        function: The wrapped view function with group access control.
    """
    
    if isinstance(group_names, str):
        group_names = [group_names] 

    def inner(view_func):
        
        @wraps(view_func)
        def _wrapper(request, *args, **kwargs):
            
            if not request.user.groups.filter(name__in=group_names).exists() and not request.user.is_superuser:
                return redirect(redirect_url)
            
            return view_func(request, *args, **kwargs)
        
        return _wrapper
    
    return inner