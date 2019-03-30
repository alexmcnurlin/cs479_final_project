#!/usr/bin/env python3

import pandas as pd
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import tensorflow as tf


df = pd.read_csv("games-features.csv")

columns_to_keep = [
    # 'QueryID',
    # 'ResponseID',
    'QueryName',
    'ResponseName',
    'ReleaseDate',
    'RequiredAge',
    'DemoCount',
    'DeveloperCount',
    'DLCCount',
    'Metacritic',
    'MovieCount',
    'PackageCount',
    'RecommendationCount',
    'PublisherCount',
    'ScreenshotCount',
    'SteamSpyOwners',
    'SteamSpyOwnersVariance',
    'SteamSpyPlayersEstimate',
    'SteamSpyPlayersVariance',
    'AchievementCount',
    'AchievementHighlightedCount',
    'ControllerSupport',
    'IsFree',
    'FreeVerAvail',
    'PurchaseAvail',
    'SubscriptionAvail',
    'PlatformWindows',
    'PlatformLinux',
    'PlatformMac',
    # 'PCReqsHaveMin',
    # 'PCReqsHaveRec',
    # 'LinuxReqsHaveMin',
    # 'LinuxReqsHaveRec',
    # 'MacReqsHaveMin',
    # 'MacReqsHaveRec',
    'CategorySinglePlayer',
    'CategoryMultiplayer',
    'CategoryCoop',
    'CategoryMMO',
    'CategoryInAppPurchase',
    'CategoryIncludeSrcSDK',
    'CategoryIncludeLevelEditor',
    'CategoryVRSupport',
    'GenreIsNonGame',
    'GenreIsIndie',
    'GenreIsAction',
    'GenreIsAdventure',
    'GenreIsCasual',
    'GenreIsStrategy',
    'GenreIsRPG',
    'GenreIsSimulation',
    'GenreIsEarlyAccess',
    'GenreIsFreeToPlay',
    'GenreIsSports',
    'GenreIsRacing',
    'GenreIsMassivelyMultiplayer',
    'PriceCurrency',
    'PriceInitial',
    'PriceFinal',
    # 'SupportEmail',
    # 'SupportURL',
    # 'AboutText',
    # 'Background',
    # 'ShortDescrip',
    # 'DetailedDescrip',
    # 'DRMNotice',
    # 'ExtUserAcctNotice',
    # 'HeaderImage',
    # 'LegalNotice',
    # 'Reviews',
    # 'SupportedLanguages',
    # 'Website',
    # 'PCMinReqsText',
    # 'PCRecReqsText',
    # 'LinuxMinReqsText',
    # 'LinuxRecReqsText',
    # 'MacMinReqsText',
    # 'MacRecReqsText'
]

df = df.drop_duplicates(["QueryName", "ResponseID"])
df = df[columns_to_keep]
df.to_csv("ml_data.csv", header=True, index=None)
